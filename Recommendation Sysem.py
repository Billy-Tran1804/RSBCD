{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2024-04-22T03:25:59.656541Z",
     "start_time": "2024-04-22T03:25:58.599324Z"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import networkx as nx\n",
    "import matplotlib.pyplot as plt\n",
    "from networkx.algorithms.community import k_clique_communities\n",
    "\n",
    "df = pd.read_json(\"/mnt/d/python/GFormer/yelp open dataset/yelp_academic_dataset_review.json\", lines=True, nrows = 100000)[[\"user_id\", \"business_id\", \"stars\"]]\n",
    "df[\"user_id\"] = \"user_\" + df[\"user_id\"]\n",
    "df['business_id'] = \"biz_\" + df['business_id']\n",
    "# Create a bipartite graph\n",
    "B = nx.Graph()\n",
    "# Add nodes with the node attribute \"bipartite\"\n",
    "B.add_nodes_from(df['user_id'].unique(), bipartite=0)  # Users\n",
    "B.add_nodes_from(df['business_id'].unique(), bipartite=1)  # Businesses\n",
    "\n",
    "# Add edges only if the rating is 4 or higher\n",
    "for _, row in df.iterrows():\n",
    "    B.add_edge(row['user_id'], row['business_id'], weight=row['stars'])\n",
    "# Project bipartite graph to one side (e.g., users)\n",
    "user_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==0}\n",
    "user_graph = nx.bipartite.weighted_projected_graph(B, user_nodes)\n",
    "\n",
    "# Apply the clique percolation method\n",
    "user_communities = list(nx.community.k_clique_communities(user_graph, 10))  # Example with k=3\n",
    "# Project bipartite graph to one side (e.g., users)\n",
    "biz_nodes = {n for n, d in B.nodes(data=True) if d['bipartite']==1}\n",
    "biz_graph = nx.bipartite.weighted_projected_graph(B, biz_nodes)\n",
    "\n",
    "# Apply the clique percolation method\n",
    "biz_communities = list(nx.community.k_clique_communities(biz_graph, 3))  # Example with k=3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "ExecuteTime": {
     "start_time": "2024-04-22T03:26:26.192763Z"
    },
    "jupyter": {
     "is_executing": true
    }
   },
   "outputs": [],
   "source": [
    "#biz ident\n",
    "df[\"rec\"] = np.empty((len(df), 0)).tolist()\n",
    "def rec_sys():\n",
    "    for usr_group in user_communities:\n",
    "        df_biz = df[df['user_id'].isin(usr_group)][\"business_id\"]\n",
    "        df_biz = pd.DataFrame(df_biz)\n",
    "        biz_com_temp = []\n",
    "        for biz_group in biz_communities:\n",
    "            for _, biz in df_biz.iterrows():\n",
    "                if biz[\"business_id\"] in biz_group:\n",
    "                    for each_biz in biz_group:\n",
    "                        biz_com_temp.append(each_biz)\n",
    "        for usr in usr_group:\n",
    "            if biz_com_temp:\n",
    "                df.at[df[df[\"user_id\"] == usr].index[0], \"rec\"] = biz_com_temp\n",
    "\n",
    "rec_sys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df[\"precision@10\"] = 0\n",
    "df[\"precision@20\"] = 0\n",
    "df[\"precision@40\"] = 0\n",
    "df[\"recall@10\"] = 0\n",
    "df[\"recall@20\"] = 0\n",
    "df[\"recall@40\"] = 0\n",
    "df[\"NDCG@10\"] = 0\n",
    "df[\"NDCG@20\"] = 0\n",
    "df[\"NDCG@40\"] = 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def metric_at_k():\n",
    "    for row_index, row in df.iterrows():\n",
    "        df_temp = df.loc[df[\"user_id\"] == df.iat[row_index,0], [\"business_id\", \"stars\"]]\n",
    "        count_gt_3 = 0\n",
    "        if len(row[\"rec\"]) >= 2:\n",
    "            print(len(row[\"rec\"]))\n",
    "            for i in range(0, 2):\n",
    "                if (df_temp[df_temp[\"business_id\"] == row[\"rec\"][i]] >=3).any():\n",
    "                    count_gt_3 +=1\n",
    "                if i == 10:\n",
    "                    #pre@10\n",
    "                    df.iat[row_index, 4] = count_gt_3/10\n",
    "                if i == 20:\n",
    "                    #pre@20\n",
    "                    df.iat[row_index, 5] = count_gt_3/20\n",
    "                if i == 40:\n",
    "                    #pre@40\n",
    "                    df.iat[row_index, 6] = count_gt_3/40\n",
    "                rec_count = 0\n",
    "                ndcg_score = 0\n",
    "                totl_rel = df_temp[\"business_id\"].count()\n",
    "                for biz in df_temp[\"business_id\"]:\n",
    "                    \n",
    "                    \n",
    "                    if biz == row[3][i]:\n",
    "                        #recall\n",
    "                        rec_count += 1\n",
    "                        print(biz, rec_count)\n",
    "                        \n",
    "                        #NDCG\n",
    "                        ndcg_score += df_temp[df_temp[\"business_id\"] == biz][\"stars\"]\n",
    "\n",
    "                    \n",
    "                    \n",
    "                    if i == 10:\n",
    "                        #recall@10\n",
    "                        if rec_count/totl_rel <= 1:\n",
    "                            df.iat[row_index, 7] = rec_count/totl_rel\n",
    "                        else:\n",
    "                            df.iat[row_index, 7] =  1\n",
    "\n",
    "                        #NDCG\n",
    "                        \n",
    "                    \n",
    "                    if i == 20:\n",
    "                        #recall@20\n",
    "                        if rec_count/totl_rel <= 1:\n",
    "                            df.iat[row_index, 8] = rec_count/totl_rel\n",
    "                        else:\n",
    "                            df.iat[row_index, 8] =  1\n",
    "                    \n",
    "                    if i == 40:\n",
    "                        #recall@40\n",
    "                        if rec_count/totl_rel <= 1:\n",
    "                            df.iat[row_index, 9] = rec_count/totl_rel\n",
    "                        else:\n",
    "                            df.iat[row_index, 9] =  1\n",
    "\n",
    "metric_at_k()\n",
    "df"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
