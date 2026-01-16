from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

import sqlite3

THRESHOLD_AMT = 500.00

class PaymentState(MessagesState):
    payment_amt: float              # Amount to be paid
    vendor_acct_num: str            # Vendor account number
    reason: str = "Payment Pending" # Payment processing status
    approved: bool = False          # Payment approved True/False
    approver: str = None            # Payment approved by
    processed: bool = False         # True once payment is processed

# Node to check if Human approval is required
def check_approval_required(state: PaymentState) -> PaymentState:
    
    print("Begining -> Approval Check Node")
    if state["payment_amt"] < THRESHOLD_AMT:
        return{"approved": True, "approver": "Automatically Approved", 
               "reason": f"Auto Approved {state['payment_amt']} under approval threshold"}
    else:
        return{"approved": False, "approver" : None, 
               "reason": f"Human Approval needed. {state['payment_amt']} higher than threshold"}


def vendor_payment_tool(payment_amt: float, vendor_acct_num: str, approved: bool=False,
                        approver: str = None) -> dict:
    """
    Process vendor payment based on approval status& logs details

    Handles both pre-approved payments& human processing decisions.

    Args: 
        param payment_amt: Description
        param vendor_acct_num: Description
        param approved: Description
        param approver: Description
    """
    print("Begining -> Vendor Payment Tool")
    if approved:
        return {"processed": True,
                "reason": f"Payment processed. Approved by {approver}"}
    else:
        return {"processed": False,
                "reason": f"Payment of {payment_amt} rejected by {approver}"}
    
# Node for HITL
def human_approval_node(state: PaymentState) -> dict:
    print("Begining -> Human Approval Node")
    if state["approved"]:
        approved=state["approved"]
        approver=state["approver"]
    else:
        value=interrupt(state)
        print(f"Human value Received: {value}")
        approved=state["approved"]
        approver=state["approver"]

        tool_call_msg= AIMessage(content="Test Msg",
                tool_calls= [
                    {"name": "vendor_payment_tool",
                     "args": {
                         "payment_amt": state["payment_amt"],
                         "vendor_acct_num" :state["vendor_acct_num"],
                         "approved" : approved,
                         "approver": approver
                     },
                     "id" : "12345"
                     }
                ])
    return {"messages": [tool_call_msg], "approved" : approved, "approver": approver}


def create_graph():
    
    payment_tool_node = ToolNode(name="tool_node", tools=[vendor_payment_tool])

    payment_graph=StateGraph(PaymentState)
    payment_graph.add_node("check_approval_required", check_approval_required)
    payment_graph.add_node("human_approval_node", human_approval_node)
    payment_graph.add_node("payment_tool", payment_tool_node)

    payment_graph.add_edge(START, "check_approval_required")
    payment_graph.add_edge("check_approval_required", "human_approval_node")
    payment_graph.add_edge("human_approval_node", "payment_tool")
    payment_graph.add_edge("payment_tool", END)

    #Checkpointer
    conn =sqlite3.connect("payment_wf_checkpoint.db", check_same_thread=False)
    sqllite_checkpointer = SqliteSaver(conn)

    payment_graph_compiled=payment_graph.compile(checkpointer=sqllite_checkpointer)

    return payment_graph_compiled

if __name__ == "__main__":
    create_graph()