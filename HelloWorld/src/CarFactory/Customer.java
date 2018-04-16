package CarFactory;

public class Customer {
    public static void main(String[] args){
        System.out.println("顾客要买BMW:");
        Car bmw = CarFactory.getCar("BMW");
        System.out.println("提取汽车："+bmw.getInfo());

        System.out.println("顾客要买BENZ");
        Car benz = CarFactory.getCar("BENZ");
        System.out.println("提取汽车："+benz.getInfo());
    }
}
