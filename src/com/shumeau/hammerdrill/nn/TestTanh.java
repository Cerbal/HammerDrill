package com.shumeau.hammerdrill.nn;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.MatrixEntry;

public class TestTanh {

	public static void main(String[] args) {
		double test_value=0.70;
		
		System.out.println(Math.tanh(test_value));
		FastTanh ft=new FastTanh();
		System.out.println(ft.tanh(test_value));
		timings();
	}
	
	public static void timings(){
		FastTanh ft=new FastTanh();
		DenseMatrix dm=Utils.createRandomMatrix(1000, 20000);
		/**
		System.out.println("apply tanh to the while matrix");
		Utils.tic();
		for(MatrixEntry me : dm){
			me.set(Math.tanh(me.get()));
		}
		Utils.toc();
		*/
		System.out.println("fastTrigo");
		Utils.tic();
		for(MatrixEntry me : dm){
			me.set(ft.tanh2(me.get()));
		}
		Utils.toc();
	}

}
