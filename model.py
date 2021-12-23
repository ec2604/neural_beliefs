import numpy as np
from modules import *
import utils


class CPCI_Action_30(nn.Module):
    def __init__(self, device, trajectory_len, negative_sampling_factor, cpc_lr, eval_lr, grid_x, grid_y, look_ahead=30):
        super(CPCI_Action_30, self).__init__()
        self.observation_to_latent_mapper = GridObservationMLP()
        self.belief_gru = beliefGRU()
        self.action_gru = actionGRU()
        self.look_ahead = look_ahead
        self.negative_sampling_factor = negative_sampling_factor
        cpc_clf = [MLP() for i in range(self.look_ahead)]
        self.mlp = nn.ModuleList(cpc_clf)
        # TODO: Use env grid size to initialize evalMLP input size
        self.eval_mlp = evalMLP((grid_x, grid_y))
        self.device = device
        self.CPC_optimizer = torch.optim.Adam(list(self.mlp.parameters()) + list(self.observation_to_latent_mapper.parameters()) + list(self.belief_gru.parameters())
                                              + list(self.action_gru.parameters()), lr=cpc_lr)
        self.position_optimizer = torch.optim.Adam(self.eval_mlp.parameters(), lr=eval_lr)
        self.trajectory_len = trajectory_len
        self.pos_cpc_tensor = None
        self.neg_cpc_tensor = None
        self.grid_x_dim = grid_x
        self.grid_y_dim = grid_y
        self.loss_cr = nn.BCELoss()
        self.loss_eval = nn.CrossEntropyLoss()

    def forward(self, b_t, a_t, z_tp1_pos, z_tp1_neg, f, init_pos, init_ori, eval_mlp=False):
        # init_pos_replicated = init_pos.view(b_t.shape[0], 1, -1)
        # init_ori_replicated = init_ori.view(b_t.shape[0], 1, -1)
        # init_ori_replicated = init_ori_replicated.repeat(1, b_t.shape[1], 1)
        # init_pos_replicated = init_pos_replicated.repeat(1, b_t.shape[1], 1)
        # Get future indices per trajectory step
        indices = torch.arange(0, self.look_ahead,device=self.device) + torch.arange(self.trajectory_len - self.look_ahead,device=self.device).view(-1, 1)
        a_for_gru = torch.gather(a_t[:, 1:, :].unsqueeze(1).expand(-1, self.trajectory_len - self.look_ahead, -1, -1), dim=2,
                     index=indices.unsqueeze(0).unsqueeze(-1).expand(b_t.shape[0], -1, -1, 4)).view(b_t.shape[0] * (self.trajectory_len - self.look_ahead), self.look_ahead, 4)

        # Generate beliefs conditioned on actions
        hidden_for_action_gru = b_t[:, :-self.look_ahead, :].reshape(-1, b_t.shape[-1])
        a_latent, _ = self.action_gru.gru1(a_for_gru, hidden_for_action_gru.unsqueeze(0))
        a_latent = a_latent.view(z_tp1_pos.shape[0],-1,self.look_ahead,b_t.shape[-1])

        # Concatenate beliefs with future z's (pos / neg)
        z_a_gru_pos = [torch.cat([z_tp1_pos[:,1:-self.look_ahead,:], a_latent[:,:-1,i,:]],dim=-1) for i in range(self.look_ahead)]
        z_a_gru_neg = [torch.cat([z_tp1_neg[:,1:-self.look_ahead,:,:], torch.unsqueeze(a_latent[:, :-1, i, :], 2).expand(-1, -1, self.negative_sampling_factor, -1)],dim=-1) for i in range(self.look_ahead)]

        orig_dim = z_a_gru_pos[0].shape
        pred_positive = torch.stack([self.mlp[i](z_a_gru_pos[i].view(-1, orig_dim[-1])).view(orig_dim[0], orig_dim[1]) for i in range(self.look_ahead)], 1)
        pred_negative = torch.stack([self.mlp[i](z_a_gru_neg[i].view(-1, orig_dim[-1])).view(orig_dim[0], orig_dim[1], self.negative_sampling_factor, -1) for i in range(self.look_ahead)], 1)
        pred_xytheta_pos, pred_xytheta_ori = None, None
        if eval_mlp:
            #pred_xytheta_pos, pred_xytheta_ori = self.eval_mlp(torch.cat([b_t.detach(), init_pos_replicated, init_ori_replicated],dim=-1).view(-1, b_t.shape[-1] + init_ori_replicated.shape[-1] + init_pos_replicated.shape[-1]))
            pred_xytheta_pos, pred_xytheta_ori = self.eval_mlp(b_t.detach().reshape(-1, b_t.shape[-1]))
            pred_xytheta_pos = pred_xytheta_pos.view(b_t.shape[0], b_t.shape[1], -1)
            pred_xytheta_ori = pred_xytheta_ori.view(b_t.shape[0], b_t.shape[1], -1)
        return pred_positive, pred_negative, (pred_xytheta_pos, pred_xytheta_ori), torch.norm(z_a_gru_pos[0].reshape(-1,1024) - z_a_gru_neg[0].reshape(-1, 1024),p=2,dim=-1).detach().cpu().numpy()

    def update(self, data_batch, save=False,  eval_mlp=False, train=True):


        obs, orig_position, orig_orientation, a_batch, new_obs, position_list, orientation_list = data_batch
        orig_position = orig_position[:, 0, ...]
        orig_orientation = orig_orientation[:, 0, ...]

        # Map observations to latents and concat with action
        z_batch = self.observation_to_latent_mapper(new_obs).view(
            len(new_obs), self.trajectory_len, -1)
        z_a_batch = torch.cat([z_batch, a_batch],dim=-1)
        # Generate beliefs
        beliefs = self.belief_gru.gru1(z_a_batch)[0]

        # Sample negatives
        z_batch_neg = utils.sample_negatives(z_batch, self.negative_sampling_factor)

        pred_positive, pred_negative, pred_xytheta, z_neg_dist = self.forward(beliefs, a_batch,
                                                                          z_batch, z_batch_neg, self.look_ahead, orig_position, orig_orientation, eval_mlp)
        if save:
            np.save('position_true.npy', position_list.detach().cpu().numpy())
            np.save('position_pred.npy', pred_xytheta[0].detach().cpu().numpy())


        if eval_mlp:
            loss_evalmlp_pos = self.loss_eval(pred_xytheta[0].view(-1, self.grid_x_dim * self.grid_y_dim),
                                              position_list.view(-1, self.grid_x_dim * self.grid_y_dim).long().max(dim=1)[1])
            loss_evalmlp_ori = self.loss_eval(pred_xytheta[1].view(-1, 4), orientation_list.view(-1, 4).long().max(dim=1)[1])
            loss_evalmlp = (loss_evalmlp_pos + loss_evalmlp_ori) / 2
            if train:
                self.position_optimizer.zero_grad()
                loss_evalmlp.backward()
                self.position_optimizer.step()

        else:
            if self.pos_cpc_tensor is None:
                self.pos_cpc_tensor = torch.ones(
                                   (pred_positive.shape[0] * pred_positive.shape[1] * pred_positive.shape[2], 1)).to(
                                   self.device)
            loss_pos = self.loss_cr(torch.sigmoid(pred_positive.reshape((-1, 1))),
                               self.pos_cpc_tensor)
            if self.neg_cpc_tensor is None:
                self.neg_cpc_tensor = torch.zeros((pred_negative.shape[0] *
                                                                                         pred_negative.shape[1] *
                                                                                         pred_negative.shape[2] *
                                                                                         pred_negative.shape[3], 1)).to(
                self.device)
            loss_neg = self.loss_cr(torch.sigmoid(pred_negative.reshape(-1, 1)), self.neg_cpc_tensor)
            loss = (loss_pos + loss_neg)
            if train:
                self.CPC_optimizer.zero_grad()
                loss.backward()
                self.CPC_optimizer.step()
        if eval_mlp:
            return loss_evalmlp_pos, loss_evalmlp_ori, pred_xytheta[0]
        else:
            return loss_pos, loss_neg, z_neg_dist
