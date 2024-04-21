# ec2_start_stop.ps1 -details
# ec2_start_stop.ps1 -start <Id>
# ec2_start_stop.ps1 -stop <Id>

param (
    [Parameter(Mandatory=$false)]
    [switch]$details,
    [Parameter(Mandatory=$false)]
    [string]$start,
    [Parameter(Mandatory=$false)]
    [string]$stop
)

# Define available options and their descriptions
$HelpOptions = @{
    details = "Details of All running EC2 instances"
    start = "start existing EC2 instance by Id"
    stop = "stop existing EC2 instance by Id"
}

# Check if help is requested
if (-not $PSScriptRoot) {
    Write-Host "This script requires arguments to function."
    Write-Host "** Remember: Only one option can be used at a time. **"
    Write-Host ""
    Write-Host "Available Options:"
    $HelpOptions.Keys | ForEach-Object { Write-Host "-$($_.PadRight(15)) : $($HelpOptions[$_])" }
    exit
  }


class EC2Instance {
    [string]$InstanceId
    [string]$Name
    [string]$InstanceType
    [string]$Status
}


function WaitForEc2State {
    param (
        [string]$state,
        [string]$instanceId
    )
    Write-Host "Waiting for state : $state"
    for ($i = 1; $i -le 20; $i++) {
        $instancesJson = aws ec2 describe-instances --instance-ids $instanceId
        $instances = $instancesJson | ConvertFrom-Json
        foreach ($instance in $instances.Reservations.Instances) {
            Write-Host "Instance: $($instance.InstanceId), $($instance.State.Name)"
        }    
        if ($($instance.State.Name) -eq $state) {
            break  
        }
        Start-Sleep -Seconds 5
    }
}

# Script logic based on provided options
if ($details) {
    Write-Host "Getting details for all available ec2 instances."
    # Get EC2 instance information (using ConvertFrom-Json for better handling)
    $instancesJson = aws ec2 describe-instances
    $instances = $instancesJson | ConvertFrom-Json
    
    # Initialize an empty list to store EC2Instance objects
    $instancesObjects = @()

    foreach ($instance in $instances.Reservations.Instances) {
        $instanceObject = [EC2Instance]::new()
        $instanceObject.InstanceId = $instance.InstanceId
        $instance_name = "UNKNOWN"
        foreach ($tag in $instance.Tags) {
            if($tag.Key -eq "Name"){
                $instance_name = $tag.Value
            }
        }
        $instanceObject.Name = $instance_name
        $instanceObject.InstanceType = $instance.InstanceType
        $instanceObject.Status = $instance.State.Name
        $instancesObjects += $instanceObject
    }

    # Print the objects (for demonstration purposes)
    $instancesObjects | Format-Table -AutoSize  

    # # Loop through each instance and get the summary
    # foreach ($reservation in $instances.Reservations) {
    #     foreach ($instanceSummary in $reservation.Instances) {                
    #         # Display the summary for each instance
    #         Write-Host "Instance: $($instanceSummary.InstanceId) \t$($instanceSummary.InstanceType) \t $($instanceSummary.KeyName) \t$($instanceSummary.State.Name)"
    #     }
    # }
}

elseif ($stop) {
    Write-Host "Stopping existing EC2 instance by Id: $stop"
    $instanceJson = aws ec2 stop-instances --instance-ids $stop
    $instance = $instanceJson | ConvertFrom-Json
    Write-Host "Command response: $($instanceSummary.instance)"
    WaitForEc2State -state "stopped" -instanceId $stop
  }
elseif ($start) {
    Write-Host "Start existing EC2 instance by Id: $start"
    $instanceJson = aws ec2 start-instances --instance-ids $start
    $instance = $instanceJson | ConvertFrom-Json
    Write-Host "Command response: $($instanceSummary.instance)"
    WaitForEc2State -state "running" -instanceId $start
  }  

