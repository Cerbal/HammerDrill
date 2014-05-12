package com.shumeau.hammerdrill.tests;

import org.junit.runner.JUnitCore;
import org.junit.runner.Result;
import org.junit.runner.notification.Failure;

public class HammerDrillTestRunner {
	  public static void main(String[] args) {
	    Result result = JUnitCore.runClasses(NeuralNetworksTest.class);
	    for (Failure failure : result.getFailures()) {
	      System.out.println(failure.toString());
	    }
	  }
}
