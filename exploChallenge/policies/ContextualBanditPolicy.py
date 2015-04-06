
class ContextualBanditPolicy:

    def getActionToPerform(self, ctx, possibleActions):
        raise NotImplementedError

    def updatePolicy(self, c, a, reward):
        raise NotImplementedError

