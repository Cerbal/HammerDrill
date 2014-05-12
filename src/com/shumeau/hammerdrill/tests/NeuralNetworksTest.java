package com.shumeau.hammerdrill.tests;

import static org.junit.Assert.*;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;

import org.junit.Test;

import com.shumeau.hammerdrill.nn.NeuralNetworks;
import com.shumeau.hammerdrill.nn.Utils;
import com.shumeau.hammerdrill.nn.Activations.Activation;
import com.shumeau.hammerdrill.nn.Evaluations.Evaluation;

public class NeuralNetworksTest {

	/**
	 * Test if the forward pass is correctly evaluated.
	 */
	@Test
	public void testForward() {
		double[][] samples={{1,1,1,1,1},{-1,-1,1,1,1},{0,0,1,1,1},{-2,1,1,1,1}};
		double[] coefficients={1,0,0,1,-0.5,-0.5,1,0.5,0.5,1,1,0, //W1
				0,0.1,0.2,  //B1
				0.1,0.2,-0.3,0.4,0.5,-0.6, //W2
				0.5,-0.5}; //B2
		double[] expected_result;
		/** at first with a linear activation */
		expected_result=new double[]{1.07,-1.88,0.47,-0.08,0.67,0.62,0.67,0.62,0.67,0.62};
		NeuralNetworks nn=new NeuralNetworks(new int[]{4,3,2}, Activation.linear);
		nn.updateCoefficient(coefficients);
		DenseMatrix result=nn.feedforward(new DenseMatrix(samples));
		double error=Utils.absDiff(result.getData(),expected_result);
		assertTrue("Result is not the one that is expected", error<0.000001);
		
		/** then with tanh */
		expected_result=new double[]{0.74931299,-0.88740211448808, 0.538359520394674,-0.328935749422230,0.42883712016634,-0.098043748057549,0.42883712016634,-0.098043748057549,0.42883712016634,-0.098043748057549};
		nn=new NeuralNetworks(new int[]{4,3,2}, Activation.tanh);
		nn.updateCoefficient(coefficients);
		result=nn.feedforward(new DenseMatrix(samples));
		error=Utils.absDiff(result.getData(),expected_result);
		assertTrue("Result is not the one that is expected", error<0.000001);
		
		/** test with softmax */
		expected_result=new double[]{0.95026348,0.04973651, 0.63413559, 0.36586441, 0.512497396, 0.4875026035,0.512497396, 0.4875026035,0.512497396, 0.4875026035};
		nn=new NeuralNetworks(new int[]{4,3,2}, new Activation[]{Activation.linear,Activation.softmax});
		nn.updateCoefficient(coefficients);
		result=nn.feedforward(new DenseMatrix(samples));
		error=Utils.absDiff(result.getData(),expected_result);
		assertTrue("Result is not the one that is expected", error<0.00001);
		
	}
	
	/**
	 * Test if the gradient is correctly evaluated.
	 */
	@Test
	public void testGradient() {
		int output_dim=20;
		int input_dim=20;
		int nb_samples=100;
		NeuralNetworks nn=new NeuralNetworks(new int[]{input_dim,40,output_dim}, Activation.linear);
		nn.initialize();
		DenseMatrix samples=(DenseMatrix) Utils.createRandomMatrix(input_dim, nb_samples);
		DenseMatrix target=(DenseMatrix) Utils.createRandomMatrix(output_dim, nb_samples);

		/** the toy coefficients */
		double[] coeffs=Utils.createRandomMatrix(nn.getTotalNumberWeights(), 1).getData();
		nn.updateCoefficient(coeffs);
		double[] gradient=nn.feedforwardAndComputeGradient(samples, target, Evaluation.mse);

		double epsilon=0.00001;
		double[] estimated_gradient=new double[gradient.length];
		for (int dim=0; dim<gradient.length; dim++){
			coeffs[dim]-=epsilon;
			nn.updateCoefficient(coeffs);
			double cost_m=nn.computeCost(samples, target, Evaluation.mse);
			coeffs[dim]+=2*epsilon;
			nn.updateCoefficient(coeffs);
			double cost_p=nn.computeCost(samples, target, Evaluation.mse);
			estimated_gradient[dim]=(cost_p-cost_m)/(2*epsilon);
			coeffs[dim]-=epsilon;
		}
		//System.out.println(Arrays.toString(estimated_gradient));
		//System.out.println(Arrays.toString(gradient));
		System.out.println(Utils.absDiff(gradient, estimated_gradient));
		System.out.println(Utils.relDiff(gradient, estimated_gradient));
		assertTrue("absolute precision is not enough high", Utils.absDiff(gradient, estimated_gradient)<1e-5);
	}
	
	/**
	 * Test if the gradient is correctly evaluated with softmax function.
	 */
	@Test
	public void testSoftmax() {
		int input_dim=20;
		int nb_samples=100;
		NeuralNetworks nn=new NeuralNetworks(new int[]{input_dim,3}, Activation.softmax);
		nn.initialize();
		DenseMatrix samples=(DenseMatrix) Utils.createRandomMatrix(input_dim, nb_samples);
		DenseMatrix target=(DenseMatrix) Matrices.random(1, nb_samples);
		for (int i=0; i<target.getData().length; i++){
			target.getData()[i]=Math.floor(target.getData()[i]*3);
		}

		/** the toy coefficients */
		double[] coeffs=Utils.createRandomMatrix(nn.getTotalNumberWeights(), 1).getData();
		nn.updateCoefficient(coeffs);
		double[] gradient=nn.feedforwardAndComputeGradient(samples, target, Evaluation.softmax);

		double epsilon=0.00001;
		double[] estimated_gradient=new double[gradient.length];
		for (int dim=0; dim<gradient.length; dim++){
			coeffs[dim]-=epsilon;
			nn.updateCoefficient(coeffs);
			double cost_m=nn.computeCost(samples, target, Evaluation.softmax);
			coeffs[dim]+=2*epsilon;
			nn.updateCoefficient(coeffs);
			double cost_p=nn.computeCost(samples, target, Evaluation.softmax);
			estimated_gradient[dim]=(cost_p-cost_m)/(2*epsilon);
			coeffs[dim]-=epsilon;
		}
		//System.out.println(Arrays.toString(estimated_gradient));
		//System.out.println(Arrays.toString(gradient));
		System.out.println(Utils.absDiff(gradient, estimated_gradient));
		System.out.println(Utils.relDiff(gradient, estimated_gradient));
		assertTrue("absolute precision is not enough high", Utils.absDiff(gradient, estimated_gradient)<1e-5);
	}
	
	/**
	 * Test if the difference between approximate tanh and exact tanh is low enough
	 */
	@Test
	public void testFastTanh() {
		DenseMatrix samples=(DenseMatrix) Utils.createRandomMatrix(720, 100);
		NeuralNetworks nn=new NeuralNetworks(new int[]{720,400,400,1000}, Activation.exact_tanh);
		nn.initialize();
		DenseMatrix with_tanh=nn.feedforward(samples);
		nn=new NeuralNetworks(new int[]{720,400,400,1000}, Activation.tanh);
		nn.initialize();
		DenseMatrix with_tanh_exact=nn.feedforward(samples);
		assertTrue("absolute precision is not enough high", Utils.absDiff(with_tanh.getData(), with_tanh_exact.getData())<2e-5);
		assertTrue("relative precision is not enough high", Utils.absDiff(with_tanh.getData(), with_tanh_exact.getData())<5e-3);
	}
	
	

}
