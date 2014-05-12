package com.shumeau.hammerdrill.tests;

import static org.junit.Assert.*;
import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.Matrices;

import org.junit.Test;

import com.shumeau.hammerdrill.nn.NeuralNetworks;
import com.shumeau.hammerdrill.nn.Utils;
import com.shumeau.hammerdrill.nn.Activations.Activation;
import com.shumeau.hammerdrill.nn.Evaluations.Evaluation;
import com.shumeau.hammerdrill.parallelization.DataContainer;
import com.shumeau.hammerdrill.parallelization.GradientParallelizer;

public class ParallelizationTest {

	/**
	 * Test if the parallel version gives the exact same result (for the gradient) as the serial one.
	 */
	@Test
	public void testParallizationGradient() {
		int nb_samples=1200;
		int samples_per_chunk=600;
		int size_input=400;
		int nb_workers=4;
		DenseMatrix samples=(DenseMatrix) Matrices.random(size_input, nb_samples);

		
		NeuralNetworks nn;
		nn=new NeuralNetworks(new int[]{size_input,200,size_input}, Activation.ReLU);
		nn.initialize();
		double[] gradient_serial=nn.feedforwardAndComputeGradient(samples, samples, Evaluation.mse);
		double cost_serial=nn.getCost();
		
		nn.initialize();
		double[] gradient_parallel=null;
		double cost_parallel=nn.getCost();
		GradientParallelizer gp=new GradientParallelizer();
		try {
			DataContainer dc=gp.parallelizeGradient(nn, nb_workers, samples_per_chunk, samples, samples, Evaluation.mse);
			gradient_parallel=dc.getGradient();
			cost_parallel=dc.getCost();
		} catch (Exception e) {
			System.out.println(e.getMessage());
			e.printStackTrace();
			fail("there has been a pb");
		}
		
		double error_gradient=Utils.absDiff(gradient_serial, gradient_parallel);
		double error_cost=Math.abs(cost_parallel-cost_serial);
		assertTrue("Gradient is not the one that is expected", error_gradient<0.000001);
		assertTrue("Cost is not the one that is expected", error_cost<0.000001);
	}

}
