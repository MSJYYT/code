package Shape;

public class Test {
    public static void main(String[] args){
        Circle circle = new Circle(1);
        System.out.println("图形名称为："+circle.getName());
        System.out.println("图形面积为："+circle.getArea());

        Rectangle rectangle = new Rectangle(1,1);
        System.out.println("图形的名称为："+rectangle.getName());
        System.out.println("图像的面积为："+rectangle.getArea());
    }
}
