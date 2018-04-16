package MyStack;

public class test {
    public static void main(String [] args){
        MyStack1 stack1 = new MyStack1();
        stack1.push(1);
        stack1.push(4);
        stack1.push(6);
        System.out.println(stack1.pop());
        System.out.println(stack1.getmin());
    }
}
