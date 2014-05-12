package com.shumeau.hammerdrill.parallelization;

import no.uib.cipr.matrix.DenseMatrix;

import com.shumeau.hammerdrill.nn.NeuralNetworks;
import com.shumeau.hammerdrill.nn.Evaluations.Evaluation;
import com.shumeau.hammerdrill.optimization.LBFGS;
import com.shumeau.hammerdrill.optimization.LBFGS.ExceptionWithIflag;

public class Trainer {
	/** The number of threads dedicated to the computation of the gradient. By default 4.
	 * It should be the number of Core that you dedicate to the task. */
	public static int nbWorkers=2;
	/** The cost function. By default mean square error */
	public static Evaluation evaluation=Evaluation.mse;
	/** The size of the chunks. By default 15000. This number must be adjusted so that the number
	 * of chunk is superior to the number of workers (e.g. 10 chunks per workers). Using a lot of chunks
	 * for each worker reduces the memory usage. However, collecting the gradient is a bottleneck. Thus,
	 * the size of the chunk must be big enough to keep the worker busy a few dozens of ms.
	 */
	public static int sizeChunks=7500;
	/**
	 * Maximum number of iteration for LBFGS. By default 500.
	 */
	public static int maxNbIteration=200;
	
	public static NeuralNetworks train(NeuralNetworks nn, DenseMatrix samples, DenseMatrix targets){
		/** realizes the first evaluation of gradient, necessary for initialization */
		GradientParallelizer gp=new GradientParallelizer();
		DataContainer dc = null;
		try {
			dc = gp.parallelizeGradient(nn, nbWorkers, sizeChunks, samples, targets, evaluation);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		
		/** variables necessary for LBFGS  */
		double[] coeffs=nn.getAllCoefficients();
		int[] flag=new int[1];
		int[] iprint=new int[3];
		iprint[0]=1;
		double[] diag=new double[nn.getTotalNumberWeights()];
		try {
			/** first time call for initialization of LBFGS */
			
			LBFGS.lbfgs(nn.getTotalNumberWeights(), 8, coeffs, dc.getCost(), dc.getGradient(), false, diag, iprint, 0.0001, 1e-10, flag);
			
			int counter=0;
			while(flag[0]==1 && counter<maxNbIteration){
				counter++;
				flag[0]=1;
				System.gc();
				LBFGS.lbfgs(nn.getTotalNumberWeights(), 8, coeffs, dc.getCost(), dc.getGradient(), false, diag, iprint, 0.0001, 1e-10, flag);
				nn.updateCoefficient(coeffs);
				dc=gp.parallelizeGradient(nn, nbWorkers, sizeChunks, samples, targets, evaluation);
			}
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.println("done");
		return nn;
	}
	
	
}
