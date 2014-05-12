package com.shumeau.hammerdrill.nn;


import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;


public class Activations {
	public enum Activation{
		tanh, exact_tanh, sigmoid, softmax, ReLU, linear;
	}
	
	/**
	 * Apply the activation.
	 * @param input input matrice
	 * @param output (SUPPOSED TO BE BLANK!!!)
	 * @param derivative (SUPPOSED TO BE BLANK!!!)
	 * @param activation
	 */
	public static void applyActivation(DenseMatrix input, DenseMatrix output, DenseMatrix derivative, Activation activation){
		switch(activation){
		case tanh: tanh(input, output, derivative); break;
		case ReLU: reLU(input, output, derivative); break;
		case linear: linear(input, output, derivative); break;
		case softmax: softmax(input, output, derivative); break;
		case exact_tanh: exact_tanh(input, output, derivative); break;
		default: System.out.println("Error, unknown activation function. By default, tanh is used"); tanh(input, output, derivative); break;
		}
	}
	
	public static void tanh(DenseMatrix input, DenseMatrix output, DenseMatrix derivative){
		FastTanh ft=new FastTanh();
		for(MatrixEntry me : input){
			double activ= ft.tanh(me.get());
			output.set(me.row(),me.column(),activ);
			derivative.set(me.row(),me.column(),ft.last_gradient);
		}
	}
	
	public static void exact_tanh(DenseMatrix input, DenseMatrix output, DenseMatrix derivative){
		for(MatrixEntry me : input){
			double activ= Math.tanh(me.get());
			output.set(me.row(),me.column(),activ);
			derivative.set(me.row(),me.column(),1-activ*activ);
		}
	}
	
	public static void reLU(DenseMatrix input, DenseMatrix output, DenseMatrix derivative){
		double temp;
		for(int i=0; i<input.getData().length;i++){
			temp=input.getData()[i];
			if (temp>0){
				output.getData()[i]=input.getData()[i];
				derivative.getData()[i]=1;
			}
		}
	}
	
	public static void linear(DenseMatrix input, DenseMatrix output, DenseMatrix derivative){
		for(int i=0; i<input.getData().length;i++){
			output.getData()[i]=input.getData()[i];
			derivative.getData()[i]=1;
		}
	}
	
	public static void softmax(DenseMatrix input, DenseMatrix output, DenseMatrix derivative){
		double[] temp=new double [input.numRows()];
		double sum=0;
		double temp_double;
		for(int i=0; i<input.numColumns();i++){
			sum=0;
			for (int j=0; j<input.numRows();j++){
				temp_double=Math.exp(input.getData()[i*input.numRows()+j]);
				temp[j]=temp_double;
				sum+=temp_double;
			}
			for (int j=0; j<input.numRows();j++){
				output.getData()[i*input.numRows()+j]=temp[j]/sum;
			}
		}
		for (int i=0; i<derivative.getData().length; i++){
			derivative.getData()[i]=1;
		}
	}
	
	public static String toString(Activation activation){
		switch(activation){
		case tanh: return "tanh";
		case ReLU: return "ReLu";
		case linear: return "linear";
		case exact_tanh: return "tanh";
		default: return "unknown activation function";
		}
	}
	
}
