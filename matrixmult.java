import java.io.File;
import java.util.Scanner;
import java.util.Arrays;

public class matrixmult{
    public static boolean canMultiply (int[][] a, int[][] b){
        if (a[0].length == b.length){
            return true;
        }
        return false;
    }

    public static int getEntry (int[] a, int[] b){
        int sum = 0;
        for (int i = 0; i < a.length, i++){
            sum += a[i] * b[i];
        }
        return sum;
    }
    public static void multiply (int[][] a, int[][] b) throws Exception{
        if (!canMultiply (a, b)){
            throw new Exception ("Incorrect dimensions.");
        }
        int[][] product = new int[a.length][b[0].length]
        for (int i = 0; i < a.length; i++){
            for (int j = 0; j < b[0].length; j++){
                int[] column = int [b.length];
                for (int )
                product[i][j] = dotProduct (a[i], )
            }
        }
        
    }


    
    public static void main (String[] args){
        int[][] matrix1 = new int[3][3];

    }

    
}
