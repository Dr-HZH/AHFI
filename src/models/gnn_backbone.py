import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, GATConv, SAGEConv, GINConv
import logging

logger = logging.getLogger(__name__)

class GNNBackbone(nn.Module):
    """GNN backbone network, supporting multiple GNN architectures"""
    
    def __init__(self, arch='GCN', nfeat=None, nhid=64, nclass=None, heads=1, dropout=0.5):
        """
        Initialize GNN backbone network
        
        Args:
            arch: GNN architecture, options: 'GCN', 'GAT', 'GraphSAGE', 'GIN'
            nfeat: Input feature dimension
            nhid: Hidden layer dimension
            nclass: Output class count
            heads: GAT head count
            dropout: Dropout probability
        """
        super(GNNBackbone, self).__init__()
        self.architecture = arch
        self.nhid = nhid
        self.heads = heads
        

        if arch == 'GCN':

            self.layer1 = GraphConv(nfeat, nhid, activation=F.relu)
            self.layer2 = GraphConv(nhid, nclass)
        
        elif arch == 'GAT':


            head_dim = nhid // heads
            

            if head_dim < 1:
                head_dim = 1
                logger.warning(f"Head count ({heads}) greater than hidden dimension ({nhid}), each head dimension set to 1")
            
            logger.info(f"GAT parameters: input_dim={nfeat}, head_output_dim={head_dim}, heads={heads}, total_output_dim={head_dim*heads}")
            

            self.layer1 = GATConv(nfeat, head_dim, heads)
            

            self.layer2 = GATConv(head_dim*heads, nclass, 1)
        
        elif arch == 'GraphSAGE':

            self.layer1 = SAGEConv(nfeat, nhid, aggregator_type='mean', activation=F.relu)
            self.layer2 = SAGEConv(nhid, nclass, aggregator_type='mean')
        
        elif arch == 'GIN':

            mlp1 = nn.Sequential(
                nn.Linear(nfeat, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nhid)
            )
            mlp2 = nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.ReLU(),
                nn.Linear(nhid, nclass)
            )
            self.layer1 = GINConv(mlp1, learn_eps=True)
            self.layer2 = GINConv(mlp2, learn_eps=True)
        
        else:
            raise ValueError(f"Unsupported GNN architecture: {arch}")
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, features):
        """
        Forward propagation
        
        Args:
            g: Graph
            features: Node features [nodes, features]
            
        Returns:
            tensor: Output features/logits [nodes, nclass]
        """

        h = self.get_first_layer_output(g, features)
        

        return self.forward_from_intermediate(g, h)
        
    def get_first_layer_output(self, g, features):
        """
        Get first layer output (intermediate representation)
        
        Args:
            g: Graph
            features: Node features [nodes, features]
            
        Returns:
            tensor: First layer output [nodes, nhid]
        """

        if self.architecture == 'GAT':

            h = self.layer1(g, features)
            

            if h.dim() > 2:
                h = h.reshape(h.size(0), -1)
                

                if h.size(1) != self.nhid:
                    if h.size(1) > self.nhid:

                        h = h[:, :self.nhid]
                    else:

                        padding = torch.zeros(h.size(0), self.nhid - h.size(1), device=h.device)
                        h = torch.cat([h, padding], dim=1)
        else:
            h = self.layer1(g, features)



        return h

    def forward_from_intermediate(self, g, h):
        """
        Continue forward propagation from intermediate representation

        Args:
            g: Graph
            h: Intermediate representation [nodes, nhid]

        Returns:
            tensor: Output features/logits [nodes, nclass]
        """

        h = self.dropout(h)


        if self.architecture == 'GAT':
            logits = self.layer2(g, h).squeeze(1)
        else:
            logits = self.layer2(g, h)
            
        return logits

    def get_representation(self, g, x):
        """Get intermediate representation (for analysis)"""
        return self.get_first_layer_output(g, x) 