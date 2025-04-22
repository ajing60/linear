public class purplecomet {
    public static void main (String[] args) {
        double num = Math.pow (199, 2);
        for (int i = 2; i < 199; i += 3) {
            double += Math.pow (i, 2) - 4 * i;
        }
        
        if (num % 2 == 0) {
            System.out.println ("2 is a factor");
        }
        for (int k =3 ;k < Math.sqrt (k);  k += 2 ){
            if (num % k == 0){
            System.out.println (k );
            num /= k;
            }
        }
    }
}
