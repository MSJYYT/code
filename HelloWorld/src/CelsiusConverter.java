import java.util.Scanner;

public class CelsiusConverter {
    public double getFahrenheit(double celsius){
        double fahrenheit = 1.8*celsius+32;
        return fahrenheit;
    }
    public static void main(String [] args){
        System.out.println("输入温度（单位：摄氏度）");
        Scanner in = new Scanner(System.in);//获得控制台输入
        double celsius = in.nextDouble();//获得用户输入的摄氏温度
        CelsiusConverter converter = new CelsiusConverter();//创建类的对象
        double fahrenheit = converter.getFahrenheit(celsius);//转换为华氏度
        System.out.println("转化为华氏度："+fahrenheit);
    }
}
