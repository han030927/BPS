import torch
from fast_pytorch_kmeans import KMeans
class NEW_Strategy:
    def __init__(self, images, net,method='kmeans'):
        self.images = images
        self.net = net
        self.method=method

    def euclidean_dist(self,x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy

        # dist.addmm_(1, -2, x, y.t())
        dist.addmm_(x,y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def query(self, n):

        embeddings = self.net.features(self.images)

        index = torch.arange(len(embeddings),device='cuda')
        if self.method == 'kmeans':
            kmeans = KMeans(n_clusters=n, mode='euclidean')
            labels = kmeans.fit_predict(embeddings)
            centers = kmeans.centroids
            dist_matrix = self.euclidean_dist(centers, embeddings)
            q_idxs = index[torch.argmin(dist_matrix, dim=1)]
        elif self.method == 'mean_feature':
            mean_tensor = torch.mean(embeddings, dim=0)
            distances = torch.norm(embeddings - mean_tensor, dim=1)
            q_idxs= torch.argsort(distances)[:100]


        return q_idxs
    
    def get_embeddings(self, images):
        embed=self.net.embed
        with torch.no_grad():
            features = embed(images).detach()
        return features

