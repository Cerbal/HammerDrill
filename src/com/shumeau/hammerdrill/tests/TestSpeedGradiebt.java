package com.shumeau.hammerdrill.tests;

import com.shumeau.hammerdrill.nn.NeuralNetworks;
import com.shumeau.hammerdrill.nn.Activations.Activation;
import com.shumeau.hammerdrill.nn.Evaluations.Evaluation;
import com.shumeau.hammerdrill.parallelization.DataContainer;
import com.shumeau.hammerdrill.parallelization.GradientParallelizer;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;

public class TestSpeedGradiebt {
	public static void main(String[] args){
		DenseMatrix samples=(DenseMatrix) Matrices.random(784, 60000);
		NeuralNetworks nn=new NeuralNetworks(new int[]{784,400,784}, Activation.tanh);
		nn.initialize();
		System.out.println("serial");
		long tic=System.currentTimeMillis();
		nn.feedforwardAndComputeGradient(samples, samples, Evaluation.mse);
		System.out.println(System.currentTimeMillis()-tic);
		
		System.out.println("parallel");
		tic=System.currentTimeMillis();
		GradientParallelizer gp=new GradientParallelizer();
		DataContainer dc = null;
		try {
			dc=gp.parallelizeGradient(nn, 4, 15000, samples, samples, Evaluation.mse);
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println(dc.getCost());
		System.out.println(System.currentTimeMillis()-tic);
		
		
	}


}
