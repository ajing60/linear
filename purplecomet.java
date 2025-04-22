public class purplecomet {
    public static void main (String[] args) {
        int num = 199^2;
        for (int i = 2; i += 3; i < 199) {
            num += i^2 - 4i;
        }
        if (num % 2 == 0) {
            System.out.println ("2 is a factor");
        }
        for (int k =2 ; k += 2; k < Math.sqrt (k)){
            if (num % k == 0){
            System.out.println (k );
            num /= k;
            }
        }
    }
}
