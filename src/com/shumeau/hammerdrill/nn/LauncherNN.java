package com.shumeau.hammerdrill.nn;


import com.shumeau.hammerdrill.data.Data;
import com.shumeau.hammerdrill.nn.Activations.Activation;
import com.shumeau.hammerdrill.nn.Evaluations.Evaluation;
import com.shumeau.hammerdrill.parallelization.Trainer;

import no.uib.cipr.matrix.DenseMatrix;


public class LauncherNN {

	public static void main(String[] args) {
		/** load the data from MNIST */
		Trainer.maxNbIteration=20;
		Debug.verbose_computation_time=false;
		
		Trainer.nbWorkers=1;
		Trainer.sizeChunks=15000;
		testMNIST();
		System.out.println(Debug.getActivationTime());
		System.out.println("done");

	}
	
	public static void testMNIST(){
		/** load the data from MNIST */
		DenseMatrix data=Data.loadFromMatlabFormat("MNIST/MNIST_train.mat", "data");
		/** corrupt 70% of the data */
		DenseMatrix corrupted_data=Data.corruptData(data, 0.7);
		
		/** define a new MLP, 784-144-784 with ReLU activation */
		NeuralNetworks nn=new NeuralNetworks(new int[]{784,144,784}, Activation.ReLU);
		nn.initialize();
		
		/** Train the network in parallel*/
		nn=Trainer.train(nn, corrupted_data, data);
		
		/** Load the test data of the MNIST, and feed the network with it */
		DenseMatrix test_data=Data.loadFromMatlabFormat("MNIST/MNIST_test.mat", "data");
		DenseMatrix output_test=nn.feedforward(test_data);
		
		/** Save the output and the neural network in matlab format for further exploitation */
		Data.saveOnMatlabFormat(output_test, "output_test.mat");
		NeuralNetworks.saveOnMatlabFormat(nn, "neural_network.mat");
		System.out.println("done");
	}
	
	public static void testMNISTClassification(){
		Trainer.evaluation=Evaluation.softmax;
		DenseMatrix data=Data.loadFromMatlabFormat("MNIST/MNIST_train.mat", "data");
		DenseMatrix targets=Data.loadFromMatlabFormat("MNIST/MNIST_train.mat", "labels");
		
		/** define a new MLP, 784-144-784 with ReLU activation */
		NeuralNetworks nn=new NeuralNetworks(new int[]{784,200,10}, new Activation[]{Activation.ReLU,Activation.softmax});
		nn.initialize();
		Utils.tic();
		/** Train the network in parallel*/
		nn=Trainer.train(nn, data, targets);
		Utils.toc();
		/** Load the test data of the MNIST, and feed the network with it */
		DenseMatrix test_data=Data.loadFromMatlabFormat("MNIST/MNIST_test.mat", "data");
		DenseMatrix output_test=nn.feedforward(test_data);
		
		/** Save the output and the neural network in matlab format for further exploitation */
		Data.saveOnMatlabFormat(output_test, "output_test.mat");
		NeuralNetworks.saveOnMatlabFormat(nn, "neural_network.mat");
		System.out.println("done");
		
		
	}

}
