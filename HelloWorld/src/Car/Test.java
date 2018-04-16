package Car;

public class Test {
    public static void main(String [] args){
        System.out.println("自定义普通汽车：");
        Car car = new Car();
        car.setName("BMW");
        car.setSpeed(100);
        System.out.println(car);
        System.out.println("自定义GPS汽车：");
        GPSCar gpsCar = new GPSCar();
        gpsCar.setName("BMW");
        gpsCar.setSpeed(100);
        System.out.println(gpsCar);
    }
}

