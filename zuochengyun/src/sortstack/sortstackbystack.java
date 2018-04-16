package sortstack;

import java.util.Random;
import java.util.Stack;

public class sortstackbystack{
    public static void SortStackbyStack(Stack<Integer> stack){
        Stack<Integer> tmp = new Stack<Integer>();
        while (!stack.isEmpty()){
            int cur = stack.pop();
            while (!tmp.isEmpty() && cur > tmp.peek()){
                stack.push(tmp.pop());
            }
            tmp.push(cur);
        }
        while (!tmp.isEmpty()){
            stack.push(tmp.pop());
        }
    }
    public static void main(String [] args){
        Stack<Integer> stack = new Stack<Integer>();
        Random ran = new Random(10);
        for (int i=0;i<10;i++){
            stack.push(ran.nextInt(10));
        }

        SortStackbyStack(stack);
        System.out.println(stack);
        System.out.println(stack.peek());
    }
}