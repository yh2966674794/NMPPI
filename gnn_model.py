from torch import scatter
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, SAGEConv, GATConv
from GCN import GCNConv
from torch_geometric.utils import to_dense_adj
from graphormer import *
from fussion import *
from torch_geometric.utils import to_dense_adj,degree, sort_edge_index,to_dense_batch
# from mamba_ssm import Mamba
from torch.nn import Dropout, Linear, Sequential
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian


def compute_laplacian_features(edge_index, k=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 构建邻接矩阵
    num_nodes = edge_index.max().item() + 1


    # 计算拉普拉斯矩阵
    laplacian_edge_index, laplacian_edge_weight = get_laplacian(edge_index, normalization='sym')

    # 创建稀疏拉普拉斯矩阵
    laplacian_matrix = torch.sparse_coo_tensor(
        indices=laplacian_edge_index,
        values=laplacian_edge_weight,
        size=(num_nodes, num_nodes),
        device=device
    )

    # 转换为稠密矩阵
    laplacian_dense = laplacian_matrix.to_dense()

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_dense)

    # 选择前k个特征向量作为位置编码
    return eigenvectors[:, :k]


def permute_within_batch(x, batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

class GIN_Net2(torch.nn.Module):
    def __init__(self, in_len=2000, in_feature=13, gin_in_feature=256, num_layers=1,
                 hidden=512, use_jk=False, pool_size=3, cnn_hidden=1, train_eps=True,
                 feature_fusion=None, class_num=7,state_size1=2,state_size2=1):
        super(GIN_Net2, self).__init__()
        self.use_jk = use_jk
        self.train_eps = train_eps
        self.feature_fusion = feature_fusion

        self.conv1d = nn.Conv1d(in_channels=in_feature, out_channels=cnn_hidden, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn_hidden)
        self.biGRU = nn.GRU(cnn_hidden, cnn_hidden, bidirectional=True, batch_first=True, num_layers=1)
        self.maxpool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        self.global_avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(math.floor(in_len / pool_size), gin_in_feature)

        self.gin_conv1 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )
        self.gin_conv2 = GINConv(
            nn.Sequential(
                nn.Linear(gin_in_feature, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ), train_eps=self.train_eps
        )

        self.linear_v = nn.Linear(512, 32)

        self.mamba1 = Mamba(seq_len=1, d_model=32, state_size=state_size1)


        self.GAT1 = pure_GAT(32,16,32,0.5,0.01,3)

        if self.use_jk:
            mode = 'cat'
            self.jump = JumpingKnowledge(mode)
            self.lin1 = nn.Linear(num_layers*hidden, hidden)
        else:
            self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

        self.lin_pre = nn.Linear(gin_in_feature+10, 32)

        self.lin3 = nn.Linear(32, 32)
        self.lin5 = nn.Linear(32, hidden)
        self.lin4 = nn.Linear(hidden, hidden)
        self.layer_norm1 = torch.nn.LayerNorm(32)

        self.layer_norm3 = torch.nn.LayerNorm(32)



        # self.fc2 = nn.Linear(hidden, class_num)

                # ---------- Edge-level fusion ----------
        self.fca = nn.Linear(hidden * 2, hidden)
        self.fca1 = nn.Linear(hidden, hidden)

        # 分类头
        self.mlp_cls = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, class_num)
        )


        self.mlp_cls = nn.Sequential(nn.Linear(32, 64), nn.ReLU(),nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32))
        # self.feature_fussion = CGAFusion(512)
        # self.feature_fussion2 = MultiModalAttention(512,8)
        # ===== 极简可学习融合参数（1 个标量）=====
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))
    def reset_parameters(self):

        self.conv1d.reset_parameters()
        self.fc1.reset_parameters()

        self.gin_conv1.reset_parameters()
        for gin_conv in self.gin_convs:
            gin_conv.reset_parameters()

        if self.use_jk:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        self.fc2.reset_parameters()


    # -------- 只改 forward --------
    def forward(self,
                x,
                edge_index,
                train_edge_id,
                p: float = 0.5,
                return_feat: bool = False):      # ← 新增开关
        x_pe = compute_laplacian_features(edge_index, k=10)

        x1, x2 = x, x                      # 两条分支的输入
        adj = to_dense_adj(edge_index)[0]

        # ---------- GIN 分支 ----------
        x1 = self.gin_conv1(x1, edge_index)

        # ---------- GAT + Mamba 全局分支 ----------
        x2 = torch.cat((x2.squeeze(-1), x_pe), 1)
        x3 = torch.cat((x.squeeze(-1), x_pe), 1)

        x2 = self.lin_pre(x2)
        x3 = self.lin_pre(x3)

        for _ in range(5):
            x2 = self.GAT1(x2, adj)
            x2 = F.dropout(x2, p=p, training=self.training)
            x2 = self.layer_norm1(x2)

            x3 = self.mamba1(x3.unsqueeze(1)).squeeze(1)
            x3 = F.dropout(x3, p=p, training=self.training)

            x2 = x2 + x3
            x2 = self.mlp_cls(x2)
            x2 = x2 + x3
            x2 = self.layer_norm3(x2)

        # ---------- 两分支各自 MLP ----------
        x1 = F.relu(self.lin1(x1))
        x1 = F.dropout(x1, p=p, training=self.training)
        x1 = self.lin2(x1)

        x2 = F.relu(self.lin5(x2))
        x2 = F.dropout(x2, p=p, training=self.training)
        x2 = self.lin4(x2)

        # ---------- Edge-level feature fusion ----------
        node_id = edge_index[:, train_edge_id]     # [2, E']

        x1_1, x1_2 = x1[node_id[0]], x1[node_id[1]]
        x2_1, x2_2 = x2[node_id[0]], x2[node_id[1]]

        x1_mul = x1_1 * x1_2                       # [E', hidden]
        x2_mul = x2_1 * x2_2                       # [E', hidden]

        # concat fusion
        x_cat = torch.cat([x1_mul, x2_mul], dim=1) # [E', 2*hidden]

        feat = F.relu(self.fca(x_cat))
        feat = F.dropout(feat, p=p, training=self.training)
        feat = self.fca1(feat)                     # [E', hidden]

        logits = self.mlp_cls(feat)                # [E', class_num]


        # ---------- 根据开关返回 ----------
        return (logits, feat) if return_feat else logits