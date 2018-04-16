package reversestack;

import java.util.Stack;

public class test {
    public static void main(String [] args){
        reverstack rstack = new reverstack();
        Stack<Integer> Stack = new Stack<Integer>();
        for (int i =1;i<6;i++){
            Stack.push(i);
        }
        System.out.println(Stack);
        //reverstack.reverse(Stack);
        rstack.reverse(Stack);
        System.out.println(Stack);

    }
}
