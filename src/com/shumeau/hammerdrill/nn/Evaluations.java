package com.shumeau.hammerdrill.nn;

import no.uib.cipr.matrix.DenseMatrix;

public class Evaluations {
	public enum Evaluation{
		mse,
		softmax;
	}
	
	/**
	 * 
	 * @param output
	 * @param target 
	 * @param derivative_error
	 * @param evaluation
	 * @return
	 */
	public static double applyEvaluation(DenseMatrix output, DenseMatrix target, DenseMatrix derivative_error, Evaluation evaluation){
		switch(evaluation){
		case mse: return mse(output, target, derivative_error);
		case softmax: return softmax(output, target, derivative_error);
		default: System.out.println("Error, unknown activation function. By default, mse is used"); return mse(output, target, derivative_error); 
		}
	}
	
	public static double mse(DenseMatrix output, DenseMatrix target, DenseMatrix derivative_error){
		assert((output.numColumns()==target.numColumns()) && (output.numRows()==target.numRows())) : "problem during evaluation of mse, the two matrices have different size";
		assert((output.numColumns()==derivative_error.numColumns()) && (output.numRows()==derivative_error.numRows())) : "problem during evaluation of mse, the two matrices have different size";

		double mse=0;
		double temp_error;
		for (int i=0; i<output.getData().length;i++){
			temp_error=output.getData()[i]-target.getData()[i];
			derivative_error.getData()[i]=temp_error;
			mse+=1.0/2*temp_error*temp_error;
		}
		mse/=(output.numColumns());
		return mse;
	}
	
	/** 
	 * Target is supposed to be a row vector, with the indice of the correct class.
	 * @param output
	 * @param target
	 * @param derivative_error
	 * @return
	 */
	public static double softmax(DenseMatrix output, DenseMatrix target, DenseMatrix derivative_error){
		if (target.numRows()!=1)
			System.out.println("There is a proble in the target format for softmax. It must be a row vector, "
					+ "with the indice of the corrcet class");
		double error=0;
		double a_t;
		int t;
		System.arraycopy(output.getData(), 0, derivative_error.getData(), 0, output.getData().length);
		for (int i=0; i<target.getData().length; i++){
			t=(int) Math.round(target.getData()[i]);
			a_t=output.get(t, i);
			error+=-Math.log(a_t);
			derivative_error.set(t, i, a_t-1);
		}
		error/=target.numColumns();
		
		return error;
	}
}
