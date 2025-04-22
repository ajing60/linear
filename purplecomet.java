public class purplecomet {
    public static void main (String[] args) {
        double num = Math.pow (2, 4);
        for (int i = 2; i < 199; i += 3) {
            num += Math.pow (i, 2) - 4 * i;
        }
        System.out.println (num);
        if (num % 2 == 0) {
            System.out.println ("2 is a factor");
            num /= 2;
        }
        System.out.println (num);
        for (int k =3 ;k < Math.sqrt (k);  k += 2 ){
            if (num % k == 0){
                System.out.println (k );
                num /= k;
            }
        }
    }
}
