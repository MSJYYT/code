package Employee;

public class Manager extends Employee {
    private double bonus;

    public double getBonus() {
        return bonus;
    }

    public void setBonus(double bonus) {
        this.bonus = bonus;
    }

    public String getInfo(){
        return "子类：我是经理";
    }
}
