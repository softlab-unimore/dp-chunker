from splitters.clause_splitter import ClauseSplitter

if __name__ == "__main__":
    sentences = [
        # "She left early because she felt sick."
        "The English professors and students protested.",
        "Young, beautiful and nice doctors and nurses worked hard.",

        # # advcl
        # "She left early because she felt sick.",
        # "He ran faster because he trained harder.",
        # "He called her when she arrived.",
        # "She studied while he slept.",
        # "The man who left because he was angry never returned.",
        # "She had a plan to escape before the guards arrived.",
        # "She thinks that he left because he was angry.",
        #
        # # acl
        # "She had a decision to make.",
        # "He found a way to escape.",
        # "He found a way to win before the time ran out.",
        # "She had something to say.",
        # "She had a plan to follow that her mentor recommended.",
        #
        # # relcl
        # "The woman who called was his sister.",
        # "The car which broke down was expensive.",
        # "The book that John wrote became famous.",
        # "The painting which the museum bought had a story to tell.",
        # "The man I met was a doctor.",
        # "The student who failed because she missed the exam asked for help.",
        # "The scientist who discovered the cure had a chance to save millions.",
        # "The book that the author who won the prize wrote inspired millions.",
        #
        # # conj
        # "He came home and took a shower.",
        # "She sat down and opened the book.",
        # "He met her at the station and he kissed her.",
        # "She cooked dinner and he washed the dishes.",
        # "He said that she left and that the door was open.",
        # "She believes that he is innocent and that the judge agreed.",
        # "He came home and said that the woman who called was his sister.",
        #
        # # ccomp
        # "He says that you like to swim.",
        # "She believes that he is innocent.",
        # "She believes that he is innocent because the evidence is clear.",
        # "He said that the woman who called was his sister.",
        # "She believes that he has a chance to win.",
        # "She knows that he believes that the earth is flat.",
        # "She left early: she was tired.",
        # "He said goodbye; he never returned.",
        #
        # # parataxis
        # "I know, I said it before.",
        # "She is smart, I believe.",
        # "He will come, I suppose.",
        # "I know, I said it before when we met.",
        # "The result was clear: the man who led the project had failed.",
        # "I know, she believes that he is innocent because the evidence was clear.",
        # "She left early: she was tired because the meeting had gone badly.",
        # "The answer was obvious: the student who studied every day passed.",
        #
        # # nominal conj
        "The professors and students protested.",
        "The doctors and nurses worked hard.",
        "The professors and students protested because the exam was unfair.",
        "The teachers and students believe that the system needs reform.",
        "The professors and students who attended protested.",
        "The professors and students who attended said that the lecture was inspiring.",
        "The professors and students protested; the dean announced that the policy would change.",
        #
        # # nominal conj con compound / named entity
        # "He met John Travolta and Bill Murray.",
        # "She interviewed Barack Obama and Joe Biden.",
        # "He read Harry Potter and Lord of the Rings.",
        # "She cited Karl Marx and Friedrich Engels and Max Weber.",
        #
        # # amod coordinati
        "American and British professors attended.",
        "Young and experienced doctors treated the patients.",
        #
        # # coordinate normali (no split atteso)
        # "She bought apples and oranges.",
        # "He met John and Mary.",
        #
        # # misti
        # "The scientist who discovered the cure said that he succeeded because he worked hard.",
        # "The man who had a plan to escape left before the guards arrived.",
        # "She told me that the doctor who treated him left because the hospital closed.",
        # "The doctors and nurses who treated him worked harder because the situation was critical.",
        # "The managers and employees said that the policy was wrong because it hurt everyone.",
        # "The book that the author who won the prize wrote inspired millions.",
        # "The scientist who discovered the cure said that he had a chance to publish because the journal that she recommended accepted his work.",
    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print(f"\n{s}")
        for split in splits:
            print(f" - {split}")