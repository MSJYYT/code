public class SumNum {
    public static void main(String[] args){
        int[] num = {1,2,3,4,5,6,7,8,9,10};
        int sum = 0;
        System.out.println("一维数组中各元素和：");
        for (int i=0;i<10;i++){
            if (i == 9){
                System.out.print(num[i]+"=");
            }else {
                System.out.print(num[i]+"+");
            }
            sum = sum+num[i];
        }
        System.out.println(sum);
    }
}
