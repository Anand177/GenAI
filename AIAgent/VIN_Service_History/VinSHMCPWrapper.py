from fastmcp import FastMCP
from BqVinOuery import (bq_vin_sh, bq_vin_service_count)

mcp=FastMCP("VIN Service History")

@mcp.tool()
def vin_service_history(vin: str):
    """
    Query Vehicle Service Information based on VIN
    
    Args:
        vin: VIN number that can be upto 17 characters long 
    
    Returns:
        Vehicle Service Information for the VIN 
    """
    try:
        data = bq_vin_sh(vin)
        return {"Status" : "Success", "data" : data}
    except Exception as e:
        return {"Status" : "Failed", "error" : str(e)}
    
@mcp.tool()
def vin_service_count(vin: str):
    """
    Query Total Repairs performed based on VIN
    
    Args:
        vin: VIN number that can be upto 17 characters long 
    
    Returns:
        VIN number, RO Number and total services performed under the VIN& RO
    """
    try:
        data = bq_vin_service_count(vin)
        return {"Status" : "Success", "data" : data}
    except Exception as e:
        return {"Status" : "Failed", "error" : str(e)}
    
if __name__ == "__main__":
    mcp.run()