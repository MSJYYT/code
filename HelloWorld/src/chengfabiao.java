public class chengfabiao {
    public static void main(String[] args){
        System.out.println("99乘法表");
        System.out.print("  ");
        //首先打印出第一行1-9
        for (int i=1;i<=9;i++)
            System.out.print(i+"           ");
        System.out.println();
        for(int i=1;i<=9;i++){
            //每一行打印出当前是第几行
            System.out.print(i+" ");
            for (int j=1;j<=9;j++)
                if(j<=i)
                    //打印出计算结果
                System.out.print(i*j+"          ");
            //每执行完一次该循环后换行
            System.out.println();
        }
    }
}
