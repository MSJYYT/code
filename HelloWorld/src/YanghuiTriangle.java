public class YanghuiTriangle {
    public static void main(String[] args){
        int triangle[][] = new int[8][];
        for (int i=0;i<triangle.length;i++){
            triangle[i] = new int[i+1];
            for (int j=0;j<triangle[i].length-1;j++){
                if (i==0 || j==0 || j==triangle[i].length-1){
                    triangle[i][j]=1;
                }else {
                    triangle[i][j]=triangle[i-1][j]+triangle[i-1][j-1];
                }
                System.out.print(triangle[i][j]+"\t");
            }
            System.out.println();
        }
    }
}
