package TwoStackqueue;

import java.util.Stack;

public class TwoStacksQueue {
    public Stack<Integer> stackPush;
    public Stack<Integer> stackPop;

    public TwoStacksQueue(){
        stackPush = new Stack<Integer>();
        stackPop = new Stack<Integer>();
    }

    public void add(int pushInt){
        stackPush.push(pushInt);
    }

    public int poll(){
        if (stackPop.empty() && stackPush.empty()){
            throw new RuntimeException("Queue is empty");
        }else if (stackPop.empty()){
            while (!stackPush.empty()){
                stackPop.push(stackPush.pop());
            }
        }
        return stackPop.pop();
    }

    public int peek(){
        if (stackPop.empty() && stackPush.empty()){
            throw new RuntimeException("Queue is empty");
        }else if (stackPop.empty()){
            while (!stackPush.empty()){
                stackPop.push(stackPush.pop());
            }
        }
        return stackPop.peek();
    }
}
