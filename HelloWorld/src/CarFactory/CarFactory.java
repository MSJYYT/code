package CarFactory;

public class CarFactory {
    public static Car getCar(String name){
        if (name.equalsIgnoreCase("BMW")){
            return new BMW();
        }else if(name.equalsIgnoreCase("BENZ")){
            return new BENZ();
        }else {
            return null;
        }
    }
}
