APPROACH:

1. First, unsupervised GraphSage is used to derive node representations
	a. Model.py has code for the unsupervised GraphSage model
	b. Loss function for unsupervised GraphSage can be found in GSUnsupLoss.py

2. Once node embeddings are derived from GraphSage, a neural network classifier is used for final binary classification task
	a. NNClassifier in Model.py file contains code for the classifier model
	b. NNClassifier can use either node embeddings from GraphSage alone or can use both, embeddings from GraphSage and given document embeddings for every node to predict existence of a link between given pair of node

3. Hyperparameter Tunig is mainly done for following parameters
	a. Learning rate for GraphSage and neural net classifier
	b. Dropout rate for GraphSage and neural net classifier

3. The model hyper parameter are set to the best combination found

4. Script start by creating
	a. Adjacency matrix from train.txtx
	b. Feature matrix from node-feat.txt
	c. Collection of negative sampling pool for every node (needed for unsupervised loss function GraphSage and in creation of training data for neural net classifier)
	d. Collection of positive sampling pool for every node (used in unsupervised loss function of GraphSage)
	e. Training/validation data for classifier, containing given links mixed with negative examples I.e. pairs of nodes without a link between them
	f. Training/validation data for GraphSage
	g. Then node embeddings are created from graphSage (which takes time)

RUNNING SCRIPT:
1. Use following command to run the code 
python predictor.py data/

2. predictions.txt file will be created in the same folder where the predictor.py file resides

3. Script also generates a folder named 'results_', this folder contains
	a. Plots for training and validation losses and accuracies




