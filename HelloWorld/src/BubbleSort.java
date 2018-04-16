public class BubbleSort {
    public static void main(String[] args){
        int[] arry=new int[]{63,4,24,1,3,13};
        System.out.println("冒泡排序过程：");
        for (int i=1;i<arry.length;i++){
            for (int j=0;j<arry.length-i;j++){
                if (arry[j]>arry[j+1]){
                    int temp=arry[j];
                    arry[j]=arry[j+1];
                    arry[j+1]=temp;
                }
                System.out.print(arry[j]+" ");
            }
            System.out.print("[");
            for (int j=arry.length-i;j<arry.length;j++){
                System.out.print(arry[j]+" ");

            }
            System.out.println("]");
        }
    }
}
