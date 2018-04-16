public class MaxNum {
    public static void main(String[] args){
        int[] num = {9,2,5,11};
        System.out.println("输出该数组：");
        for (int i=0;i<num.length;i++){
            System.out.print(num[i]+"、");
        }
        int max = 0;
        for (int j=0;j<num.length;j++){
            if (max < num[j]){
                max = num[j];
            }
        }
        System.out.println("数组中最大值:"+max);
    }
}
