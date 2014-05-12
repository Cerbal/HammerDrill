package com.shumeau.hammerdrill.parallelization;

import com.shumeau.hammerdrill.nn.NeuralNetworks;
import com.shumeau.hammerdrill.nn.Evaluations.Evaluation;

import no.uib.cipr.matrix.DenseMatrix;

public class GradientWorker implements Runnable {
	DataContainer data_container;
	NeuralNetworks nn;
	DenseMatrix samples, target;
	Evaluation evaluation;
	int low_bound, high_bound;
	
	/**
	 * Constructor of the gradient worker
	 * @param dc the data container that will be used to put the computed gradient.
	 * @param samples a pointer to the whole set of pointer.
	 * @param target a pointer to the whole set of target.
	 * @param low_bound defines the beginning of the segment of data treated by this worker. Worker takes care of all data : low_bound<=data<upper_bound.
	 * @param upper_bound defines the end of the segment of data treated by this worker. Worker takes care of all data : low_bound<=data<upper_bound.
	 * @param evaluation the evaluation method used
	 * @param nn The neural network to optimise.
	 */
	GradientWorker(DataContainer dc, DenseMatrix samples, DenseMatrix target, int low_bound, int upper_bound, Evaluation evaluation, NeuralNetworks nn){
		this.nn=nn;
		this.samples=samples;
		this.target=target;
		this.evaluation=evaluation;
		this.data_container=dc;
		this.high_bound=upper_bound;
		this.low_bound=low_bound;
	}
	
	@Override
	public void run() {
		//long tic=System.currentTimeMillis();
		/** extract the portion of data that this thread works on */
		DenseMatrix chunk=new DenseMatrix(samples.numRows(), this.high_bound-this.low_bound);
		System.arraycopy(samples.getData(), low_bound*samples.numRows(), chunk.getData(), 0, (high_bound-low_bound)*samples.numRows());
		DenseMatrix target_chunk=new DenseMatrix(target.numRows(), this.high_bound-this.low_bound);
		System.arraycopy(target.getData(), low_bound*target.numRows(), target_chunk.getData(), 0, (high_bound-low_bound)*target.numRows());
		//long toc=System.currentTimeMillis();
		//System.out.println(toc-tic);
		/** compute the gradient */
		double[] gradient=nn.feedforwardAndComputeGradient(chunk, target_chunk, evaluation);
		double cost=nn.getCost();
		

		
		/** give the computed information to the data container */
		data_container.updateData(gradient,cost,chunk.numColumns());
		

	}

}
