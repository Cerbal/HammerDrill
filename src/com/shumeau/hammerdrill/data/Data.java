package com.shumeau.hammerdrill.data;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.shumeau.hammerdrill.nn.Utils;

import no.uib.cipr.matrix.DenseMatrix;

public class Data {
	private static class MatrixWrapper implements Serializable{
		private static final long serialVersionUID = -6744806164439303893L;
		int rows,column;
		double[] data;
		
		public MatrixWrapper(int rows, int column, double[] data) {
			this.rows=rows;
			this.column=column;
			this.data=data;
		}
		
		public DenseMatrix toMatrix(){
			DenseMatrix dm=new DenseMatrix(rows,column);
			System.arraycopy(data, 0, dm.getData(), 0, dm.getData().length);
			return dm;
		}
	}
	
	public static void saveOnDisk(DenseMatrix dm, String path) {
	    ObjectOutputStream obj_out;
	    MatrixWrapper mw=new MatrixWrapper(dm.numRows(), dm.numColumns(), dm.getData());
		try {
		    FileOutputStream f_out = new FileOutputStream(path);
			obj_out = new ObjectOutputStream (f_out);
		    obj_out.writeObject (mw);
		    obj_out.flush();
		    f_out.flush();
		    obj_out.close();
		    f_out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static DenseMatrix loadFromDisk(String path){
		MatrixWrapper matrix=null;
		try {
			FileInputStream f_in = new FileInputStream(path);
		    ObjectInputStream obj_in = new ObjectInputStream (f_in);
		    matrix= (MatrixWrapper) obj_in.readObject();
		    obj_in.close();
		    f_in.close();
		} catch (IOException | ClassNotFoundException e) {
			e.printStackTrace();
		}
	    return matrix.toMatrix();
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static void saveOnMatlabFormat(DenseMatrix dm, String path) {
		 MLDouble mlDouble = new MLDouble( "matrix", dm.getData(), dm.numRows() );
		 ArrayList list = new ArrayList<Object>();
		 list.add( mlDouble );
		 
		 try {
			new MatFileWriter( path, list );
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static DenseMatrix loadFromMatlabFormat(String path, String arrayName) {
		 MatFileReader matFileReader;
		try {
			Utils.tic();
			 matFileReader = new MatFileReader(path);
			 MLArray mla=matFileReader.getContent().get(arrayName);
			 Utils.toc();
			 Utils.tic();
			 MLDouble mld=(MLDouble) mla;
			 DenseMatrix dm=new DenseMatrix(mld.getM(),mld.getN());
			 mld.getArray(dm.getData());
			 /**
			 double[][] data_tab=mld.getArray();
			 for (int i=0; i<data_tab.length; i++){
				 for (int j=0; j<data_tab[0].length; j++){
					 dm.getData()[i*data_tab[0].length+j]=data_tab[i][j];
				 }
			 }
			 */
			 Utils.toc();
			 
			 return dm;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;

	}
	
	public static DenseMatrix corruptData(DenseMatrix dm, double proportion) {
		DenseMatrix dm2=new DenseMatrix(dm.numRows(), dm.numColumns());
		for (int i=0; i<dm.getData().length; i++){
			if (Math.random()>proportion)
				dm2.getData()[i]=dm.getData()[i];
		}
		return dm2;
	}
}
