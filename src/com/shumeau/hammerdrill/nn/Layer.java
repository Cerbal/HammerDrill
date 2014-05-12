package com.shumeau.hammerdrill.nn;

import java.util.ArrayList;

import no.uib.cipr.matrix.DenseMatrix;

public abstract class Layer {
	/**
	 * Return the output of the layer and store it into A.
	 * @param X
	 */
	public abstract DenseMatrix feedforward(DenseMatrix X);
	
	/**
	 * Compute the gradient. Store it into internal variables. To get the gradient, call
	 * packGradient.
	 * @param R_l derivative toward the output of the layer.
	 * @param A_l_1 input of the layer.
	 * @return The derivative towards the input of the layer (R_l_1)
	 */
	abstract DenseMatrix computeGradient(DenseMatrix R_l, DenseMatrix A_l_1);
	
	/** Pack the coefficients of the gradient into a double array.*/
	abstract double[] packGradient();
	/** Pack the coefficients of the layer into a double array.*/
	abstract double[] packCoefficients();
	/** Pack the coefficients of the layer into a double array.*/
	abstract void unpackCoefficients(double[] coefs);
	
	/** Get the total number of weights of the layer. */
	abstract int getNumberWeights();
	/** Pack the coefficients of the layer into a double array.*/
	abstract DenseMatrix getOutput();
	
	/** get some characteristic about the layer */
	abstract int getInSize();
	abstract int getOutSize();
	
	/** return a minimal clone : only the structure is copied, not the coefficients */
	abstract Layer cloneStructure();
	/** return a full clone : coefficients W and B are cloned (but not the temporary variables such  */
	abstract Layer cloneCoefs();
	
	/** Initialize the network */
	abstract void initialize();
	
	/** should help to clear the unused memory */
	abstract void clear();
	
	/** fill the arraylist to store into matlab file */
	abstract void toMatlabFormat(@SuppressWarnings("rawtypes") ArrayList arrayList,int id_layer);
	
	
}
