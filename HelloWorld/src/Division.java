public class Division {
    public static void main(String[] args){
        String s = "无言独上西楼，月如钩，寂寞梧桐深院锁清秋。"+
                    "剪不断，理还乱，是离愁，别是一般滋味在心头";
        System.out.println("原字符串长度："+s.length());
        String[] news = s.split("，|。");
        for (int i=0;i<news.length;i++){
            System.out.println(news[i]);
        }
        System.out.println("分行后字符串数组的长度："+news.length);
    }
}
