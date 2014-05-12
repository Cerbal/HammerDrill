package com.shumeau.hammerdrill.nn;


import java.util.Random;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;

public class Utils {
	static long tic_time;
	static int nb_op=0;
	public static final boolean verbose=false;
	
	public static DenseMatrix createRandomMatrix(int rows, int cols){
		Random gen=new Random(1);
		double factor=Math.sqrt(cols);
		DenseMatrix A= new DenseMatrix(rows, cols);
		for(MatrixEntry me : A){
			me.set(gen.nextGaussian()/factor);
		}
		return A;
	}
	public static void tic(){
		tic_time=System.currentTimeMillis();
	}
	public static void toc(){
		double toc_time=System.currentTimeMillis();
		System.out.println("ellapsed time : "+(toc_time-tic_time));
	}
	
	/**
	 * plot dense matrix.
	 */
	public static void showMatrix(DenseMatrix md, String label){
		System.out.println("Matrix: "+label);
			for (int j=0; j<md.numRows(); j++){
				for (int i=0; i<md.numColumns(); i++){
				System.out.print(md.get(j, i)+" ");
			}
			System.out.println("");
		}
	}
	
	/**
	 * Get the absolute difference between two array of double.
	 */
	public static double absDiff(double[] d1, double[] d2){
		assert (d1.length==d2.length) : "the two arrays do not have the same size";
		double max_diff=0;
		for (int j=0; j<d1.length; j++){
			if (Math.abs(d1[j]-d2[j])>max_diff) max_diff=Math.abs(d1[j]-d2[j]);
		}
		return max_diff;
	}
	
	/**
	 * Get the relative difference between two array of double.
	 */
	public static double relDiff(double[] d1, double[] d2){
		assert (d1.length==d2.length) : "the two arrays do not have the same size";
		double max_diff=0;
		for (int j=0; j<d1.length; j++){
			if ((Math.abs(d1[j]-d2[j])/d1[j])>max_diff) max_diff=Math.abs(d1[j]-d2[j])/d1[j];
		}
		return max_diff;
	}
	
}
