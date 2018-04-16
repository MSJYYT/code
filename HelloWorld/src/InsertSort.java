public class InsertSort {
    public static void main(String[] args){
        int arry[] = {4,3,5,9,2,6};
        for (int i=0;i<arry.length;i++){
            System.out.print(arry[i]+" ");
        }
        System.out.println("插入排序后：");
        int tmp;
        int j;
        for (int i = 1;i<arry.length;i++){
            j = i;
            tmp = arry[i];
            while (j > 0 && arry[j-1] > tmp){
                arry[j] = arry[j-1];
                j--;
            }
            arry[j] = tmp;
        }
        for (int i=0;i<arry.length;i++){
            System.out.print(arry[i]+" ");
        }
    }
}
