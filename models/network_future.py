import torch
import torch.nn as nn
from IPython import embed
import math
from utils import utils_transform
from models.video_transformer import SpaceTimeTransformer

nn.Module.dump_patches = True


class SlowFastForecastHead(nn.Module):
    def __init__(
            self,
            embed_dim,
            out_dim,
            hidden_dim=256,
            alpha=4,
            beta=4,
            dropout=0.0,
            recent_frames=0,
            use_recent_delta=False,
            video_dim=0):
        super(SlowFastForecastHead, self).__init__()
        self.alpha = max(1, int(alpha))
        self.recent_frames = max(0, int(recent_frames))
        self.use_recent_delta = bool(use_recent_delta)
        fast_dim = max(8, embed_dim // max(1, int(beta)))
        slow_dim = embed_dim
        recent_dim = embed_dim * self.recent_frames
        if self.use_recent_delta:
            recent_dim += embed_dim

        self.fast_proj = nn.Linear(embed_dim, fast_dim)
        self.slow_proj = nn.Linear(embed_dim, slow_dim)
        self.fast_pathway = nn.Sequential(
                            nn.Conv1d(fast_dim, fast_dim, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv1d(fast_dim, fast_dim, kernel_size=3, padding=1),
                            nn.ReLU()
        )
        self.slow_pathway = nn.Sequential(
                            nn.Conv1d(slow_dim + fast_dim, slow_dim, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv1d(slow_dim, slow_dim, kernel_size=3, padding=1),
                            nn.ReLU()
        )
        self.fast_pool = nn.AdaptiveAvgPool1d(1)
        self.slow_pool = nn.AdaptiveAvgPool1d(1)

        output_layers = [
                            nn.Linear(slow_dim + fast_dim + recent_dim + video_dim, hidden_dim),
                            nn.ReLU()
        ]
        if dropout > 0:
            output_layers.append(nn.Dropout(dropout))
        output_layers.append(nn.Linear(hidden_dim, out_dim))
        self.output = nn.Sequential(*output_layers)

    def _slow_sample(self, x):
        return x[:, ::self.alpha, :]

    def _recent_summary(self, encoded_sequence):
        if self.recent_frames <= 0 and not self.use_recent_delta:
            return None

        summaries = []
        if self.recent_frames > 0:
            recent = encoded_sequence[:, -self.recent_frames:, :]
            if recent.shape[1] < self.recent_frames:
                pad = recent[:, :1, :].expand(-1, self.recent_frames - recent.shape[1], -1)
                recent = torch.cat([pad, recent], dim=1)
            summaries.append(recent.reshape(recent.shape[0], -1))

        if self.use_recent_delta:
            if encoded_sequence.shape[1] > 1:
                delta = encoded_sequence[:, -1, :] - encoded_sequence[:, -2, :]
            else:
                delta = torch.zeros_like(encoded_sequence[:, -1, :])
            summaries.append(delta)

        return torch.cat(summaries, dim=1)

    def forward(self, encoded_sequence, video_features=None):
        fast_sequence = self.fast_proj(encoded_sequence)
        fast_features = self.fast_pathway(fast_sequence.permute(0, 2, 1))

        slow_sequence = self.slow_proj(self._slow_sample(encoded_sequence))
        fast_lateral = self._slow_sample(fast_features.permute(0, 2, 1))
        slow_input = torch.cat([slow_sequence, fast_lateral], dim=2)
        slow_features = self.slow_pathway(slow_input.permute(0, 2, 1))

        fast_summary = self.fast_pool(fast_features).squeeze(-1)
        slow_summary = self.slow_pool(slow_features).squeeze(-1)
        summary = torch.cat([slow_summary, fast_summary], dim=1)
        recent_summary = self._recent_summary(encoded_sequence)
        if recent_summary is not None:
            summary = torch.cat([summary, recent_summary], dim=1)
        if video_features is not None:
            summary = torch.cat([summary, video_features], dim=1)
        return self.output(summary)




class EgoCast(nn.Module):
    def __init__(self, input_dim, output_dim, num_layer, embed_dim, nhead, device,opt):
        super(EgoCast, self).__init__()
        self.window_size = opt['datasets']['train']['window_size']
        self.linear_embedding = nn.Linear(input_dim,embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)        

        
        self.include_video = opt['netG']['video_model']
        if opt['datasets']['train']['use_aria']:
            if opt['datasets']['train']['use_rot']:
                out_num = 7 if opt['datasets']['train']['output']=='aria' else 58#70
                out_dim = out_num*opt['datasets']['train']['future_frames']
            else:
                out_num = 3 if opt['datasets']['train']['output']=='aria' else 54#60
                out_dim =  out_num*opt['datasets']['train']['future_frames']
        else:
            out_dim =63*opt['datasets']['train']['future_frames']
        self.forecast_head_type = opt['netG'].get('forecast_head', 'legacy')
        if self.include_video:
            self.video_model = SpaceTimeTransformer(num_frames=opt['datasets']['train']['window_size']+1)
            if self.forecast_head_type == 'slowfast':
                self.forecast_head = SlowFastForecastHead(
                                embed_dim=embed_dim,
                                out_dim=out_dim,
                                hidden_dim=opt['netG'].get('slowfast_hidden_dim', 256),
                                alpha=opt['netG'].get('slowfast_alpha', 4),
                                beta=opt['netG'].get('slowfast_beta', 4),
                                dropout=opt['netG'].get('slowfast_dropout', 0.0),
                                recent_frames=opt['netG'].get('slowfast_recent_frames', 0),
                                use_recent_delta=opt['netG'].get('slowfast_use_recent_delta', False),
                                video_dim=768)
            else:
                self.ap = nn.AdaptiveAvgPool1d(embed_dim)
                self.stabilizer = nn.Sequential(
                                nn.Linear((embed_dim)+768, 256),
                                nn.ReLU(),
                                nn.Linear(256, out_dim)
                )
            
        else:
            if self.forecast_head_type == 'slowfast':
                self.forecast_head = SlowFastForecastHead(
                                embed_dim=embed_dim,
                                out_dim=out_dim,
                                hidden_dim=opt['netG'].get('slowfast_hidden_dim', 256),
                                alpha=opt['netG'].get('slowfast_alpha', 4),
                                beta=opt['netG'].get('slowfast_beta', 4),
                                dropout=opt['netG'].get('slowfast_dropout', 0.0),
                                recent_frames=opt['netG'].get('slowfast_recent_frames', 0),
                                use_recent_delta=opt['netG'].get('slowfast_use_recent_delta', False))
            else:
                self.stabilizer = nn.Sequential(
                                nn.AdaptiveAvgPool1d(embed_dim),
                                nn.Linear(embed_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, out_dim)
                )
            self.joint_rotation_decoder = nn.Sequential(
                             nn.Linear(embed_dim, 256),
                             nn.ReLU(),
                             nn.Linear(256, 126)
             )

        if self.forecast_head_type not in ['legacy', 'slowfast']:
            raise NotImplementedError('Forecast head [{:s}] is not found.'.format(self.forecast_head_type))

        #self.body_model = body_model

    def forward(self, input_tensor,image=None, do_fk = True):

        input_tensor = input_tensor.reshape(input_tensor.shape[0],input_tensor.shape[1],-1)
        x = self.linear_embedding(input_tensor)
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        if self.include_video:
            a = self.video_model(image)
            if self.forecast_head_type == 'slowfast':
                global_orientation = self.forecast_head(x, a)
            else:
                x = x.reshape(x.shape[0],-1)
                x = self.ap(x)
                x_mixed = torch.cat([x,a],axis=1)
                global_orientation = self.stabilizer(x_mixed)
        else:
            if self.forecast_head_type == 'slowfast':
                global_orientation = self.forecast_head(x)
            else:
                x = x.reshape(x.shape[0],-1)
                global_orientation = self.stabilizer(x)
        return global_orientation
