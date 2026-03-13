from splitters.clause_splitter import ClauseSplitter


if __name__ == '__main__':

    sentences = [
    #
    #     # advcl con advmod annidato
    #     "She left early because she felt sick.",
    #     "He ran faster because he trained harder.",
    #
    #     # advcl con relcl annidato
    #     "He succeeded because the mentor who guided him was excellent.",
    #     "She failed because the machine that she used was broken.",
    #
    #     # advcl con acl annidato
    #     "He won because he had a strategy to follow.",
    #     "She cried because she had no reason to stay.",
    #
    #     # ccomp con advcl annidato
    #     "She thinks that he left because he was angry.",
    #     "He believes that she stayed because she loved him.",
    #
    #     # ccomp con relcl annidato
    #     "He said that the woman who called was his sister.",
    #     "She knows that the car that he bought was expensive.",
    #
    #     # ccomp con acl annidato
    #     "He said that she had a plan to escape.",
    #     "She believes that they found a cure to test.",
    #
    #     # relcl con advcl annidato
    #     "The man who left because he was angry never returned.",
    #     "The student who failed because she missed the exam asked for help.",
    #
    #     # acl con advcl annidato
    #     "She had a plan to escape before the guards arrived.",
    #     "He found a way to win before the time ran out.",
    #
    #     # conj con ccomp
    #     "He said that she left and that the door was open.",
    #     "She believes that he is innocent and that the judge agreed.",
    #
    #     # nominal conj con advcl
    #     "The professors and students protested because the exam was unfair.",
    #     "The doctors and nurses worked hard because the situation was critical.",
    #
    #     # nominal conj con ccomp
    #     "The teachers and students believe that the system needs reform.",
    #     "The managers and employees agreed that the policy was wrong.",
    #
    #     # nominal conj con relcl + ccomp
    #     "The professors and students who attended said that the lecture was inspiring.",
    #     "The doctors and nurses who worked said that the conditions were difficult.",
    #
    #     # tutto insieme
    #     "The manager who hired him said that he had a chance to succeed because the team that she built was strong.",
    #     "The teacher who failed her believed that she had a reason to appeal because the exam that he wrote was unfair.",
    #
        # # relcl + advcl + ccomp
        # "The scientist who discovered the cure said that he succeeded because he worked hard.",
        # "The teacher who failed her believed that she cheated because the answers were identical.",
        #
        # # relcl + acl + advcl
        # "The man who had a plan to escape left before the guards arrived.",
        # "The woman who found a way to survive waited until the storm passed.",
        #
        # # ccomp + relcl + advcl
        # "She told me that the doctor who treated him left because the hospital closed.",
        # "He said that the lawyer who defended her won because the evidence was clear.",
        #
        # # nominal conj + relcl + advcl
        # "The doctors and nurses who treated him worked harder because the situation was critical.",
        # "The professors and students who attended protested because the rules were unfair.",
        #
        # # nominal conj + ccomp + advcl
        # "The managers and employees said that the policy was wrong because it hurt everyone.",
        # "The teachers and parents believed that the system failed because nobody acted.",
        #
        # # ccomp + ccomp + relcl
        # "She knows that he believes that the woman who called was lying.",
        # "He said that she thinks that the man who left never returned.",
        #
        # # relcl + relcl + ccomp
        # "The book that the author who won the prize wrote inspired millions.",
        # "The cure that the scientist who worked alone discovered saved thousands.",
        #
        # # advcl + acl + ccomp
        # "He succeeded because he had a strategy to follow that his mentor recommended.",
        # "She failed because she missed a chance to prepare that her teacher offered.",
        #
        # # conj + ccomp + relcl
        # "He came home and said that the woman who called was his sister.",
        # "She left early and told me that the man who followed her was dangerous.",
        #
        # # tutto insieme
        # "The scientist who discovered the cure said that he had a chance to publish because the journal that she recommended accepted his work.",
        # "The teacher who inspired her believed that she had a reason to succeed because the school that he founded supported its students.",
        # "The managers and employees who attended said that they found a way to solve the problem because the consultant that the board hired was excellent.",

        # # dep_=parataxis esplicito
        # "I know, I said it before.",
        # "She is smart, I believe.",
        # "He will come, I suppose.",
        # "It was wrong, I admit.",
        #
        # # disguised parataxis con ":"
        # "She left early: she was tired.",
        # "The result was clear: they had failed.",
        # "He made his choice: he resigned.",
        # "The answer is simple: nobody came.",
        #
        # # disguised parataxis con ";"
        # "He said goodbye; he never returned.",
        # "She worked hard; the results showed it.",
        # "The door was open; someone had broken in.",
        #
        # # parataxis + altre subordinate
        # "She left early: she was tired because the meeting had gone badly.",
        # "I know, I said it before when we met.",
        # "The result was clear: the man who led the project had failed.",

        # # parataxis esplicito con subordinate annidate
        # "I know, I said it before because I was angry.",
        # "She is smart, I believe, because she solved the problem that nobody could fix.",
        # "He will come, I suppose, if the weather is good.",
        # "It was wrong, I admit, because the man who led the project never consulted anyone.",
        #
        # # disguised parataxis con ":" + subordinate
        # "The result was clear: the team that worked hardest had won because they prepared well.",
        # "She made her decision: she would leave because the company that hired her had changed.",
        # "The answer was obvious: the student who studied every day passed because he never gave up.",
        # "He understood the truth: the woman he loved had left because he never listened.",
        #
        # # disguised parataxis con ";" + subordinate
        # "She worked hard; the results showed it because the numbers never lie.",
        # "He said goodbye; he never returned because the city that raised him had changed.",
        # "The door was open; someone had broken in before the guard who patrolled arrived.",
        #
        # # parataxis + ccomp
        # "I know, she believes that he is innocent because the evidence was clear.",
        # "He admitted it; she thinks that the man who confessed was lying because he was scared.",
        #
        # # parataxis + nominal conj
        # "The professors and students protested; the dean announced that the policy would change.",
        # "I know, the teachers and parents believe that the system failed because nobody acted.",

        # Base cases già testati
        "He came home, took a shower and immediately went to bed.",
        "He met her at the station and he kissed her.",
        "American and British professors and students are very good.",
        "American and British professors and students who love something are very good.",
        "The book that John wrote became famous because it inspired many readers.",
        "The woman who looked happy danced when the music started.",
        "The scientist who discovered the cure had a chance to save millions.",
        "The painting which the museum bought had a story to tell.",
        "She had a decision to make because her boss resigned.",
        "He found a way to escape before the door closed.",
        "The man I met introduced me to the woman who won the prize.",
        "The book that she wrote inspired the student who solved the problem.",
        "The scientist who discovered the cure had a chance to publish because the journal accepted his work.",
        "The movie that we watched had a scene to remember because it moved everyone.",

        # ccomp
        "He says that you like to swim.",
        "She believes that he is innocent because the evidence is clear.",
        "He said that the book that John wrote was boring.",
        "She believes that he has a chance to win.",
        "He says that you like to swim and she likes to dance.",
        "She believes that he is innocent and that the trial was unfair.",
        "She knows that he believes that the earth is flat.",
        "She believes that they found a cure to test.",
        "She believes that he had a chance to win because he trained hard.",
        "The professors and students believe that the exam was unfair.",
        "The scientist who discovered the cure said that he had a chance to publish because the journal that she recommended accepted his work.",

        # parataxis
        "I know, I said it before.",
        "She is smart, I believe.",
        "She left early: she was tired.",
        "The result was clear: they had failed.",
        "He said goodbye; he never returned.",
        "I know, I said it before when we met.",
        "She left early: she was tired because the meeting had gone badly.",
        "The result was clear: the man who led the project had failed.",
        "I know, she believes that he is innocent because the evidence was clear.",
        "The answer was obvious: the student who studied every day passed because he never gave up.",
        "He understood the truth: the woman he loved had left because he never listened.",

        # nominal conj + parataxis/subordinate
        "The professors and students protested; the dean announced that the policy would change.",
        "The professors and students protested because the exam was unfair.",
        "The doctors and nurses who treated him worked harder because the situation was critical.",
        "The managers and employees said that the policy was wrong because it hurt everyone.",

        # relcl annidate
        "The book that the author who won the prize wrote inspired millions.",
        "The cure that the scientist who worked alone discovered saved thousands.",
        "The man who left because he was angry never returned.",
        "The woman who looked happy danced when the music started.",

        # complessi misti
        "He came home and said that the woman who called was his sister.",
        "She thinks that he left because he was angry.",
        "She believes that he had a chance to win because he trained hard.",
        "The door was open; someone had broken in before the guard who patrolled arrived.",
        "The scientist who discovered the cure said that he is innocent and that the trial was unfair.",
        "She knows that he believes that the woman who called was lying.",
    ]

    splitter = ClauseSplitter()

    for s in sentences:
        splits = splitter.split_sentence(s)
        print("\nSentence:", s)
        for split in splits:
            print(" -", split)