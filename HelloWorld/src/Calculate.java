public class Calculate {
    final float PI=3.14159f;
    public float getArea(float r){
        float area = PI*r*r;
        return area;
    }

    public float getArea(float l,float w){
        float area = l*w;
        return area;
    }

    public void draw(int num){
        System.out.println("画"+num+"个任意形状的图形");
    }

    public void draw(String shape){
        System.out.println("画一个"+shape);
    }

    public static void main(String[] args){
        Calculate calculate = new Calculate();
        float l = 20;
        float w = 30;
        float areaRectangle = calculate.getArea(l,w);
        System.out.println("长为"+l+"宽为"+w+"的面积为"+areaRectangle);

        float r=7;
        float areaCirc = calculate.getArea(r);
        System.out.println("半径为"+r+"的圆的面积为"+areaCirc);

        int num = 7;
        calculate.draw(num);
        calculate.draw("三角形");
    }
}
