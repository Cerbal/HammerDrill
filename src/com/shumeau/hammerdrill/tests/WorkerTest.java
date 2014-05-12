package com.shumeau.hammerdrill.tests;

public class WorkerTest implements Runnable {
		double[] tab;
		int start, end;
		@Override
		public void run()  {
			
			try {
				Thread.sleep(1);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			for (int i=start;i<end;i++)
				tab[i]=Math.tanh(i*i);
			
			
		}
			


}
