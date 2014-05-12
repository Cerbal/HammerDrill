package com.shumeau.hammerdrill.nn;

public class Debug {
	public static boolean verbose_computation_time=false;
	private static long activationTime=0;
	public static synchronized void addActivationTime(int time){
		activationTime+=time;
	}
	
	public static synchronized long getActivationTime(){
		return activationTime;
	}
}
