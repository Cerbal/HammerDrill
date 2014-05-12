package com.shumeau.hammerdrill.tests;


import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;


public class TestParallelizationMJT {

	public static void main(String[] args) {
		
		ExecutorService executor=Executors.newFixedThreadPool(4);	
		int size=10000000;
		double[] tab=new double[6*size];
		long tic=System.currentTimeMillis();
		for(int i=0; i<6;i++){
			WorkerTest wt=new WorkerTest();
			wt.start=i*10000000;
			wt.end=(i+1)*10000000;
			wt.tab=tab;
			executor.execute(wt);
		}
		executor.shutdown();
		try {
			executor.awaitTermination(1000, TimeUnit.SECONDS);
		} catch (InterruptedException e) {e.printStackTrace();}
		System.out.println(System.currentTimeMillis()-tic);
	}

}
