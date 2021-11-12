APPROACH:

1. Supervised GraphSage algorithm is used to derive node embeddings
	a. GSage.py has model code

2. Hyperparameter Tunis is done for following parameters
	a. Depth for GraphSage
	b. Hidden layer dimensions for (1, depth)-hop distance layers
	c. Dropout rate for every hidden layer
	d. Sample size parameter of GraphSage for each k-hop distance neighborhood
	e. Learning rate

3. The model hyper parameter are set to the best combination found

4. Script start by creating
	a. Adjacency matrix
	b. BERT representations for each node-title - THIS TAKES 15 MINUTES


RUNNING SCRIPT:
1. Use following command to run the code 
python classifier.py network.txt categories.txt titles.txt train.txt val.txt test.txt

2. predictions.txt file will be created in the same folder where the classifier.py file resides

3. Script also generates a folder named 'results_', this folder contains
	a. Plots for training and validation losses and accuracies
	b. PCS visualization of the node embeddings from each layer of GNN



