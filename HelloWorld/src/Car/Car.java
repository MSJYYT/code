package Car;

public class Car {
    private String name;
    private double speed;

    public String getName(){
        return name;
    }

    public void setName(String name){
        this.name = name;
    }

    public double getSpeed(){
        return speed;
    }

    public void setSpeed(double speed){
        this.speed = speed;
    }
    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("车名："+name+",");
        sb.append("速度："+speed+"千米/小时");
        return sb.toString();
    }
}
