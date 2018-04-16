package DogandCatQueue;

public class Pet {
    private String type;

    public Pet(String type){
        this.type = type;
    }

    public String getPetType(){
        return this.type;
    }
}

class Dog extends Pet{
    public Dog(){
        super("dog");
    }
}

class Cat extends Pet{
    public Cat(){
        super("cat");
    }
}